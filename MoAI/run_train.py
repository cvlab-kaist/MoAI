import argparse
import logging
import math
import os
import os.path as osp
from os.path import join
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import wandb
import imageio
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import torch.utils.checkpoint
import transformers
import cv2
# import open3d as o3d
import multiprocessing as mp
import inspect

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
import random
import webdataset as wds
import time
from PIL import Image
from time import gmtime, strftime


from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from genwarp.models.resnet import InflatedConv3d, InflatedGroupNorm
from genwarp.models.mutual_self_attention import ReferenceAttentionControl
from genwarp.models.pose_guider import PoseGuider
from genwarp.models.unet_2d_condition import UNet2DConditionModel
from genwarp.models.unet_3d import UNet3DConditionModel
from genwarp.models.hook import UNetCrossAttentionHooker, XformersCrossAttentionHooker
from genwarp.pipelines.pipeline_nvs import NVSPipeline
from training_utils import delete_additional_ckpt, import_filename, seed_everything, load_model
from einops import rearrange

from training_utils import forward_warper, camera_controller, plucker_embedding, get_embedder, get_coords
from torchvision.transforms.functional import to_pil_image

from lora_diffusion import inject_trainable_lora, extract_lora_ups_down

# from dust3r.inference import inference
# from dust3r.model import AsymmetricCroCo3DStereo
# from dust3r.utils.image import load_images
# from dust3r.image_pairs import make_pairs
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

# from inference_utils import batchify_known

from train_model import embedding_prep
from train_marigold import mari_embedding_prep
from run_duster import make_video

import copy

# from genwarp.utils.projector import ndc_rasterizer

from genwarp.utils import (
    reprojector,
    ndc_rasterizer,
    get_rays,
    pose_to_ray,
    one_to_one_rasterizer,
    mesh_rendering,
    features_to_world_space_mesh,
    torch_to_o3d_mesh,
    torch_to_o3d_cuda_mesh,
    compute_plucker_embed,
    postprocess_co3d,
    postprocess_realestate,
    postprocess_combined,
    postprocess_vggt,
    PointmapNormalizer,
    UncertaintyLoss
)

from data.co3d_dataset import Co3DDataset
from realestate_dataset import RealEstateDataset

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from functools import partial

# real10k
from dataset.preprocessed_re10k import PreprocessedRe10k
# megascenes
from dataset.preprocessed_megascenes import PreprocessedMegaScenes

from transformers import CLIPImageProcessor
from training_utils import pred_depth_inference

from dataset.scannet import Scannetdataset

import lpips  # pip install lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        geometry_unet: UNet3DConditionModel,
        geometry_unet_2,
        geometry_unet_3,
        geo_reference_unet,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
        geo_reference_control_writer = None,
        geo_reference_control_reader = None,
        inference = False,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.geometry_unet = geometry_unet
        self.geometry_unet_2 = geometry_unet_2
        self.geometry_unet_3 = geometry_unet_3
        self.geo_reference_unet = geo_reference_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.geo_reference_control_writer = geo_reference_control_writer
        self.geo_reference_control_reader = geo_reference_control_reader
        self.inference = inference
        self.cfg_guidance_scale = 3.5

    def forward(
        self,
        noisy_latents,
        geo_noisy_latents,
        geo_noisy_latents_2,
        geo_noisy_latents_3,
        timesteps,
        ref_image_latents,
        geo_ref_latents,
        clip_image_embeds,
        ref_coord_embeds,
        tgt_coord_embed,
        uncond_fwd: bool = False,
        correspondence = None,
        weight_dtype = None,
        gt_target_coord_embed = None,
        task_emb = None,
        attn_proc_hooker=None,
        geo_attn_proc_hooker=None,
        geo_attn_proc_hooker_2=None,
        geo_attn_proc_hooker_3=None,
        closest_idx=None,
        val_scheduler=None,
        vae=None
    ): 
                
        ref_cond_latents = []
        num_viewpoints = len(ref_coord_embeds)

        for ref_embed in ref_coord_embeds:
            ref_cond_tensor = ref_embed.to(device="cuda").to(weight_dtype).unsqueeze(2)
            ref_cond = self.pose_guider(ref_cond_tensor)
            
            ref_cond_latents.append(ref_cond[:,:,0,...])
        
        tgt_cond_tensor = tgt_coord_embed.to(device="cuda").unsqueeze(2)
        tgt_cond_latent = self.pose_guider(tgt_cond_tensor)
        tgt_cond_latent = tgt_cond_latent[:,:,0,...]
        batch_size = tgt_cond_latent.shape[0]

        if not uncond_fwd:
            if not self.inference:
                ref_timesteps = torch.zeros_like(timesteps)
            else:
                ref_timesteps = torch.zeros(batch_size * 2).to(ref_image_latents.device)
                uncond_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
                
                clip_image_embeds = torch.cat(
                    [uncond_image_prompt_embeds, clip_image_embeds], dim=1
                )
                
                ref_image_latents = ref_image_latents.repeat(1,2,1,1,1)

                if self.geo_reference_unet is not None:
                    geo_ref_latents = geo_ref_latents.repeat(1,2,1,1,1)
                ref_cond_latents = [ref.repeat(2,1,1,1) for ref in ref_cond_latents]
                tgt_cond_latent = tgt_cond_latent.repeat(2,1,1,1)
                                
            for i, ref_latent in enumerate(ref_image_latents):
                self.reference_unet(
                    ref_latent,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds[i],
                    pose_cond_fea=ref_cond_latents[i],
                    return_dict=False,
                    reference_idx=i,
                )
                
                if self.geo_reference_unet is not None:
                    self.geo_reference_unet(
                        geo_ref_latents[i],
                        ref_timesteps,
                        encoder_hidden_states=clip_image_embeds[i],
                        pose_cond_fea=ref_cond_latents[i],
                        return_dict=False,
                        reference_idx=i,
                    )

            self.reference_control_reader.update(self.reference_control_writer, correspondence=correspondence)
            
            if self.geo_reference_unet is not None:
                self.geo_reference_control_reader.update(self.geo_reference_control_writer, correspondence=correspondence)
        
        if closest_idx is not None:
            clip_closest_embeds = []
            for batch_num in range(clip_image_embeds.shape[1]):
                clip_closest_embeds.append(clip_image_embeds[closest_idx[batch_num], batch_num])
            tgt_clip_embed = torch.stack(clip_closest_embeds)
        else:
            tgt_clip_embed = clip_image_embeds[0]

        if not self.inference:
            model_pred = self.denoising_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=tgt_clip_embed,
                pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                class_labels=task_emb
            ).sample
        
            self.reference_control_reader.clear()
            self.reference_control_writer.clear()
            
            if attn_proc_hooker is not None:        
                for k, tensor_list in attn_proc_hooker.image_attention_dict.items():
                    geo_attn_proc_hooker.image_attention_dict[k] = tensor_list.copy()
                    
                    if self.geometry_unet_2 is not None:
                        geo_attn_proc_hooker_2.image_attention_dict[k] = tensor_list.copy()
                    
                    if self.geometry_unet_3 is not None:
                        geo_attn_proc_hooker_3.image_attention_dict[k] = tensor_list.copy()

                geo_attn_proc_hooker.layer_list = attn_proc_hooker.layer_list.copy()
            
            if self.geometry_unet_2 is not None:
                geo_attn_proc_hooker_2.layer_list = attn_proc_hooker.layer_list.copy()

            if self.geometry_unet_3 is not None:
                geo_attn_proc_hooker_3.layer_list = attn_proc_hooker.layer_list.copy()
                
            if self.geometry_unet is not None:                 
                geo_model_pred = self.geometry_unet(
                    geo_noisy_latents,
                    timesteps,
                    encoder_hidden_states=tgt_clip_embed,
                    pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                    class_labels=task_emb
                ).sample
                        
                results_dict = {
                    "img_pred": model_pred,
                    "geo_pred": geo_model_pred
                }

            else:
                results_dict = {
                    "img_pred": model_pred,
                }
            
            if self.geometry_unet_2 is not None:
                geo_model_pred_2 = self.geometry_unet_2(
                    geo_noisy_latents_2,
                    timesteps,
                    encoder_hidden_states=tgt_clip_embed,
                    pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                    class_labels=task_emb
                ).sample

                geo_attn_proc_hooker_2.clear()
                
                results_dict["geo_pred_2"] = geo_model_pred_2

            if self.geometry_unet_3 is not None:
                geo_model_pred_3 = self.geometry_unet_3(
                    geo_noisy_latents_3,
                    timesteps,
                    encoder_hidden_states=tgt_clip_embed,
                    pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                    class_labels=task_emb
                ).sample

                geo_attn_proc_hooker_3.clear()
                
                results_dict["geo_pred_3"] = geo_model_pred_3
                
            if self.geo_reference_unet is not None:
                self.geo_reference_control_reader.clear()
                self.geo_reference_control_writer.clear()

            if attn_proc_hooker is not None:
                attn_proc_hooker.clear()
                geo_attn_proc_hooker.clear()
                    
        else: # Inference code
            extra_step_kwargs = prepare_extra_step_kwargs(val_scheduler)
            
            input_dict = {
                "img_pred": noisy_latents,
                "geo_pred": geo_noisy_latents,
                "geo_pred_2": geo_noisy_latents_2 if self.geometry_unet_2 is not None else None,
                "geo_pred_3": geo_noisy_latents_3 if self.geometry_unet_3 is not None else None
            }
            
            warped_image_latents = geo_noisy_latents[:,:4]
             
            for n, t in tqdm(enumerate(timesteps)):
                results_dict = {}
                
                noisy_latents = input_dict["img_pred"]
                latent_model_input = torch.cat([noisy_latents] * 2)
                latent_model_input = val_scheduler.scale_model_input(
                        latent_model_input, t
                    )
                                
                geo_noisy_latents = input_dict["geo_pred"]
                # add warped_image_latent
                if n != 0:
                    geo_noisy_latents = torch.cat([warped_image_latents, geo_noisy_latents], dim=1)
                geo_latent_model_input = torch.cat([geo_noisy_latents] * 2)
                geo_latent_model_input = val_scheduler.scale_model_input(
                        geo_latent_model_input, t
                    )
                
                if self.geometry_unet_2 is not None:
                    geo_noisy_latents_2 = input_dict["geo_pred_2"]
                    if n != 0:
                        geo_noisy_latents_2 = torch.cat([warped_image_latents, geo_noisy_latents_2], dim=1)
                    geo_latent_model_input_2 = torch.cat([geo_noisy_latents_2] * 2)
                    geo_latent_model_input_2 = val_scheduler.scale_model_input(
                            geo_latent_model_input_2, t
                        )

                if self.geometry_unet_3 is not None:
                    geo_noisy_latents_3 = input_dict["geo_pred_3"]
                    if n != 0:
                        geo_noisy_latents_3 = torch.cat([warped_image_latents, geo_noisy_latents_3], dim=1)
                    geo_latent_model_input_3 = torch.cat([geo_noisy_latents_3] * 2)
                    geo_latent_model_input_3 = val_scheduler.scale_model_input(
                            geo_latent_model_input_3, t
                        )
                
                model_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=tgt_clip_embed,
                    pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                    return_dict=False,
                    class_labels=task_emb
                )[0]
                
                for k, tensor_list in attn_proc_hooker.image_attention_dict.items():
                    geo_attn_proc_hooker.image_attention_dict[k] = tensor_list.copy()
                    
                    if self.geometry_unet_2 is not None:
                        geo_attn_proc_hooker_2.image_attention_dict[k] = tensor_list.copy()
                    
                    if self.geometry_unet_3 is not None:
                        geo_attn_proc_hooker_3.image_attention_dict[k] = tensor_list.copy()

                geo_attn_proc_hooker.layer_list = attn_proc_hooker.layer_list.copy()
                
                if self.geometry_unet_2 is not None:
                    geo_attn_proc_hooker_2.layer_list = attn_proc_hooker.layer_list.copy()

                if self.geometry_unet_3 is not None:
                    geo_attn_proc_hooker_3.layer_list = attn_proc_hooker.layer_list.copy()
                            
                geo_model_pred = self.geometry_unet(
                    geo_latent_model_input,
                    t,
                    encoder_hidden_states=tgt_clip_embed,
                    pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                    return_dict=False,
                    class_labels=task_emb
                )[0]

                attn_proc_hooker.clear()
                geo_attn_proc_hooker.clear()
                        
                results_dict = {
                    "img_pred": model_pred,
                    "geo_pred": geo_model_pred
                }
                
                if self.geometry_unet_2 is not None:
                    geo_model_pred_2 = self.geometry_unet_2(
                        geo_latent_model_input_2,
                        t,
                        encoder_hidden_states=tgt_clip_embed,
                        pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                        return_dict=False,
                        class_labels=task_emb
                    )[0]

                    geo_attn_proc_hooker_2.clear()
                    
                    results_dict["geo_pred_2"] = geo_model_pred_2

                if self.geometry_unet_3 is not None:
                    geo_model_pred_3 = self.geometry_unet_3(
                        geo_latent_model_input_3,
                        t,
                        encoder_hidden_states=tgt_clip_embed,
                        pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                        return_dict=False,
                        class_labels=task_emb
                    )[0]

                    geo_attn_proc_hooker_3.clear()
                    
                    results_dict["geo_pred_3"] = geo_model_pred_3
                
                for key in results_dict.keys():
                    noise_pred = results_dict[key]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    
                    if key == "img_pred":
                        prev = input_dict[key]
                    else:
                        if n == 0:
                            prev = input_dict[key][:, 4:]
                        else:
                            prev = input_dict[key]
                        
                    latents_noisy = val_scheduler.step(
                        noise_pred, t, prev, **extra_step_kwargs,
                        return_dict=False
                    )[0]
                    
                    input_dict[key] = latents_noisy
            
            fin_results_dict = {}
            for key, latent in input_dict.items():
                if latent is not None:
                    latent = latent.squeeze(2)
                    synthesized = decode_latents(vae, latent)
                    fin_results_dict[key] = synthesized
            
            results_dict = fin_results_dict

            self.reference_control_reader.clear()
            self.reference_control_writer.clear()
                        
        return results_dict

def decode_latents(
    vae,
    latents,
    normalize=True
):
    latents = 1 / 0.18215 * latents
    rgb = []
    for frame_idx in range(latents.shape[0]):
        rgb.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)

    rgb = torch.cat(rgb)
        
    if normalize:
        rgb = (rgb / 2 + 0.5).clamp(0, 1)
    return rgb.squeeze(2)


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    
    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def apply_heatmap(tensor):
    # Ensure tensor is in the right format (1, 1, 512, 512)
    assert tensor.ndim == 4 and tensor.shape[1] == 1, "Tensor must have shape (1, 1, H, W)"
    
    # Remove batch dimension and convert to numpy array
    image_np = tensor[0, 0].cpu().numpy()
    
    # Normalize the tensor to range 0-255 for visualization
    image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply heatmap (COLORMAP_AUTUMN) using OpenCV
    heatmap_np = cv2.applyColorMap(image_np, cv2.COLORMAP_MAGMA)
    
    # Convert back to tensor and add batch dimension
    heatmap_tensor = T.ToTensor()(heatmap_np).unsqueeze(0)
    heatmap_tensor = torch.stack((heatmap_tensor[:,2],heatmap_tensor[:,1],heatmap_tensor[:,0]),dim=1)
    
    return heatmap_tensor


def convert_depth_to_normal(depth: torch.Tensor) -> None:
    """
    Converts a depth map tensor (values 0-1) to a normal map image and saves it.
    
    Args:
        depth_map (torch.Tensor): A tensor of shape (H, W) with depth values in [0, 1].
        output_path (str): The path to save the normal map image file.
    """
    device = depth.device
    H, W = depth.shape[-2], depth.shape[-1]

    depth = (depth + 1) * 127.5

    # Define Sobel kernels for x and y gradients (shape: (1,1,3,3))
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)
    
    # Compute the gradients using convolution (pad=1 to maintain spatial dimensions)
    dx = F.conv2d(depth, sobel_x, padding=1)
    dy = F.conv2d(depth, sobel_y, padding=1)

    # Compute the normal vectors:
    # For each pixel, normal = (-dx, -dy, 1)
    ones = torch.ones_like(dx)
    normal = torch.cat((-dx, -dy, ones), dim=1)
    
    # Normalize the normal vectors
    norm = torch.sqrt((normal ** 2).sum(dim=1, keepdim=True))
    # Avoid division by zero: use torch.where to replace zeros with ones
    normal = normal / torch.where(norm != 0, norm, torch.ones_like(norm))

    return normal

def prepare_extra_step_kwargs(
    scheduler,
    generator=None,
    eta = 0.0
):
    accepts_eta = 'eta' in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta

    # check if the scheduler accepts generator
    accepts_generator = 'generator' in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs['generator'] = generator
    return extra_step_kwargs


def prepare_duster_embedding(
    src_images,
    correspondence,
    camera_info,
    embedder=None,
    src_idx=None,
    tgt_idx=None,
    dataset="co3d",
    projection_config="target_only",  
    normalize_coord=True,
    tgt_image=None,
    point_masking=False,
    masking_percent=0.5,
    use_gt_cor=False,
    pure_gt=False,
    gt_threshold=0.005,
    ref_depth=None,
    confidence=None,
    warp_image=True,
    mesh_pts=None,
    mesh_depth=None,
    mesh_normals=None,
    mesh_ref_normals=None,
    normal_mask=None,
    plucker=None,
    gt_cor_regularize=False,
    pts_norm_func=None,
    current_dataset="multi",
    full_condition=False,
    nomesh_normal=False
):
    
    # Prepare inputs.
    src_image = src_images[0]
    B = src_image.shape[0] # Image shape (B, num_views, 3, H, W)
    num_ref_views = correspondence["ref"].shape[1]
    H, W = src_image.shape[-2:]
    device = src_image.device 

    depth_conditioning = True if ref_depth is not None else False
    conf_conditioning = True if confidence is not None else False
    norm_conditioning = True if mesh_ref_normals is not None else False
    plucker_conditioning = True if plucker is not None else False
    
    if mesh_pts is None or current_dataset == "realestate":
        pts_rgb = torch.cat(src_images,dim=1).permute(0,1,3,4,2).reshape(B,-1,3)
        pts_locs = correspondence["ref"].reshape(B,-1,3)
        
        if normalize_coord:
            if pts_norm_func is not None and current_dataset != "realestate":
            # if pts_norm_func is not None:
                src_corr = pts_norm_func(correspondence["ref"].to(device))
                
                if full_condition:
                    full_proj_results = pts_norm_func(correspondence["tgt"].to(device))
    
            else:
                max = torch.max(pts_locs, dim=-2)[0]
                min = torch.min(pts_locs, dim=-2)[0]
                src_corr = ((correspondence["ref"] - min[:,None,None,None,:]) / (max[:,None,None,None,:] - min[:,None,None,None,:])) * 2 - 1
                
                if full_condition:
                    full_proj_results = ((correspondence["tgt"] - min[:,None,None,:]) / (max[:,None,None,:] - min[:,None,None,:])) * 2 - 1
            
            mod_pts_locs = src_corr.reshape(B,-1,3)

        else:
            mod_pts_locs = pts_locs

        if not full_condition:
            # Warping Image
            if warp_image and conf_conditioning:
                conf = confidence[...,None].reshape(B,-1,1)
                pts_feat = torch.cat((mod_pts_locs, pts_rgb, conf), dim=-1)    
            elif conf_conditioning:
                conf = confidence[...,None].reshape(B,-1,1)
                pts_feat = torch.cat((mod_pts_locs, conf), dim=-1)
            elif warp_image:
                pts_feat = torch.cat((mod_pts_locs, pts_rgb), dim=-1)
            else:
                pts_feat = mod_pts_locs

            # if correspondence["ref"] is None:
            if point_masking:
                num_points = pts_locs.shape[1]

                masked_pts = int((1 - masking_percent) * num_points)
                pts_idx = torch.randperm(num_points)[:masked_pts]

                pts_locs = pts_locs[:,pts_idx]
                pts_feat = pts_feat[:,pts_idx]

            coord_channel = pts_feat.shape[-1]
            camera_tgt = camera_info["tgt"]

            combined_results, tgt_depth = reprojector(pts_locs, pts_feat, camera_tgt, device=device, coord_channel=coord_channel, get_depth=depth_conditioning)
            proj_results = combined_results[...,:3]
            
            # predict mesh normals
            if current_dataset == "realestate" and nomesh_normal:
                variance_filter = filter_high_variance_regions(tgt_depth)
                normal = convert_depth_to_normal(tgt_depth)
                
                mesh_normals = (variance_filter * normal).permute(0,2,3,1)
                
            if normal_mask is not None and current_dataset != "realestate":
                proj_results = normal_mask * proj_results
                tgt_depth = normal_mask.permute(0,3,1,2) * tgt_depth

            if warp_image and conf_conditioning:
                warped_results = combined_results[...,3:6]
                tgt_conf = combined_results[...,6:]
            elif conf_conditioning:
                tgt_conf = combined_results[...,3:]
                warped_results = combined_results[...,4:]
            else:
                warped_results = combined_results[...,3:]
                                                
        else:
            proj_results = full_proj_results
            tgt_depth = correspondence["tgt"][...,2:].permute(0,3,1,2)
            warped_results = tgt_image.permute(0,2,3,1)
                    
        camera_ref = {}
        camera_ref["pose"] = camera_info["ref"]["pose"][:,0]        
        camera_ref["focals"] = camera_info["ref"]["focals"][:,0]
    
    fin_embed = embedder(src_corr).permute(1,0,4,2,3)
    tgt_embed = embedder(proj_results).permute(0,3,1,2)
        
    # Conditions.
    tgt_mask = (proj_results[...,0][...,None] == 0).float().permute(0,3,1,2)
    full_mask = torch.zeros_like(tgt_mask, device=device)

    src_loc_embeds = []
    
    if depth_conditioning:
        ref_depth = ref_depth.permute(1,0,4,2,3)
    if conf_conditioning:
        confidence = confidence.unsqueeze(2).permute(1,0,2,3,4)
    if norm_conditioning:
        ref_norm = mesh_ref_normals.permute(1,0,2,3,4)
    if plucker_conditioning:
        ref_plucker = plucker["ref"].permute(1,0,2,3,4)
        tgt_plucker = plucker["tgt"].squeeze(1)
    
    for i, emb in enumerate(fin_embed):
        ref_catlist = [emb]
        if norm_conditioning:
            ref_catlist += [ref_norm[i]]
        if depth_conditioning:
            ref_catlist += [ref_depth[i]]
        if plucker_conditioning:
            ref_catlist += [ref_plucker[i]]
        if conf_conditioning:
            ref_catlist += [confidence[i]]
        ref_catlist +=  [full_mask]
        cat_emb = torch.cat(ref_catlist, dim=1)
        src_loc_embeds.append(cat_emb)
        
    tgt_catlist = [tgt_embed]
        
    if norm_conditioning:
        tgt_catlist += [mesh_normals.permute(0,3,1,2)]
    if depth_conditioning:
        tgt_catlist += [tgt_depth]
    if plucker_conditioning:
        tgt_catlist += [tgt_plucker]
    if conf_conditioning:
        tgt_conf = tgt_conf.permute(0,3,1,2)
        tgt_catlist += [tgt_conf]
    tgt_catlist += [tgt_mask]

    tgt_loc_embed = torch.cat(
        tgt_catlist, dim=1)

    conditions = dict(
        ref_embeds=src_loc_embeds,
        tgt_embed=tgt_loc_embed,
        ref_correspondence=src_corr,
        gt_tgt_embed=None
    )

    # Outputs.
    renders = dict(
        warped=warped_results.permute(0,3,1,2),
        correspondence=proj_results.permute(0,3,1,2),
        tgt_depth=tgt_depth
        
    )

    return conditions, renders


def convert_opencv_extrinsics_to_view(R_cv: torch.Tensor, t_cv: torch.Tensor) -> torch.Tensor:
    """
    Converts OpenCV-style extrinsics [R|t] to a 4x4 view matrix using a look-at formulation.
    The look-at convention used here computes:
      f = normalize(eye - target)
      l = normalize(cross(up, f))
      u = normalize(cross(f, l))
    and forms the view matrix as M = [R | -R*eye] (in 4x4 form).
    
    OpenCV camera coordinate system (x-right, y-down, z-forward) is assumed.
    
    Args:
        R_cv (torch.Tensor): 3x3 rotation matrix from OpenCV.
        t_cv (torch.Tensor): 3-element translation vector from OpenCV.
        
    Returns:
        torch.Tensor: 4x4 view matrix in the look-at convention.
    """
    # Compute the camera center in world coordinates:
    eye = -R_cv.t() @ t_cv  # C = -R^T * t

    # In OpenCV, the camera looks along the positive z-axis.
    # Define target as eye + (R_cv^T * [0,0,1])
    forward_cv = R_cv.t() @ torch.tensor([0.0, 0.0, 1.0], dtype=R_cv.dtype, device=R_cv.device)
    target = eye + forward_cv

    # Define up using the camera's up direction from OpenCV:
    up = R_cv.t() @ torch.tensor([0.0, 1.0, 0.0], dtype=R_cv.dtype, device=R_cv.device)

    # Compute the look-at basis vectors:
    f = F.normalize(eye - target, dim=0)       # Forward vector (points from target to eye)
    l = F.normalize(torch.cross(up, f), dim=0)   # Left vector (perpendicular to up and f)
    u = F.normalize(torch.cross(f, l), dim=0)    # Recomputed up vector

    # Assemble the rotation matrix (using rows: left, up, forward)
    R_lookat = torch.stack([l, u, f], dim=0)  # 3x3 rotation

    # Build the 4x4 view matrix:
    M_view = torch.eye(4, dtype=R_cv.dtype, device=R_cv.device)
    M_view[:3, :3] = R_lookat
    # The translation part is given by -R_lookat * eye
    M_view[:3, 3] = -R_lookat @ eye

    return M_view


def make_video(frame_list, now, output_folder = "outputs/", folder_name=None): 
    samples = torch.stack(frame_list)
    vid = (
        (samples.permute(0,2,3,1) * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    new_dir = output_folder + f"{now}/{folder_name}"
    os.makedirs(new_dir, exist_ok=True)

    video_path = os.path.join(new_dir, "video.gif")

    # imageio.mimwrite(video_path, vid)
    imageio.mimsave(video_path, vid, 'GIF', fps=1)

    for i, image in enumerate(samples):
        save_image(image, new_dir + f"/frame_{i}.png")



def find_closest_camera(reference_cameras: torch.Tensor, target_pose: torch.Tensor):
    """
    Compares a set of reference camera poses to a target pose and returns the index
    of the closest reference camera based on the Frobenius norm of the difference.

    Args:
        reference_cameras (torch.Tensor): Tensor of shape (B, N, 4, 4), where B is the batch size,
                                            and N is the number of reference cameras.
        target_pose (torch.Tensor): Tensor of shape (B, 4, 4) representing the target camera pose.
    
    Returns:
        int: The index of the closest reference camera (for the first batch element).
    """
    # Expand target_pose to shape (B, 1, 4, 4) so that broadcasting works with reference_cameras (B, N, 4, 4)
    ref_origins = reference_cameras[:, :, :3, -1]
    # For target poses: shape (B, 3)
    target_origins = target_pose[:, :3, -1]
    
    # Expand target_origins to (B, 1, 3) for broadcasting against each reference origin in the same batch.
    target_origins_expanded = target_origins.unsqueeze(1)
    
    # Compute Euclidean distances along the last dimension (axis=2) for each reference camera.
    distances = torch.norm(ref_origins - target_origins_expanded, dim=2)  # shape: (B, N)
    
    # For each batch element, get the index of the reference camera with the smallest distance.
    closest_indices = torch.argmin(distances, dim=1)

    return closest_indices

def depth_normalize(cfg, depth):
    t_min = torch.tensor(cfg.depth_min, device=depth.device)
    t_max = torch.tensor(cfg.depth_max, device=depth.device)

    normalized_depth = (((depth - t_min) / (t_max - t_min)) - 0.5 ) * 2.0

    return normalized_depth


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def encode_depth(depth, vae, weight_dtype):
    # Depth: (B, H, W, 1)

    normalized_depth = depth_normalize(depth)
    stacked_depth = normalized_depth.repeat(1,1,1,3).permute(0, 3, 1, 2)
    
    latent_depth = vae.encode(stacked_depth.to(weight_dtype)).latent_dist.sample()

    return latent_depth


def filter_high_variance_regions(depth: torch.Tensor,
                                 kernel_size: int = 3,
                                 threshold: float = 0.01) -> torch.Tensor:
    """
    Zeroes out pixels in a depth map whose local variance exceeds `threshold`.

    Args:
        depth (torch.Tensor): (B, 1, H, W) depth tensor.
        kernel_size (int): Neighborhood size for variance (must be odd).
        threshold (float): Variance threshold.

    Returns:
        torch.Tensor: Filtered depth of same shape (B, 1, H, W).
    """
    B, C, H, W = depth.shape
    assert C == 1, "Expected depth shape (B,1,H,W)"
    pad = kernel_size // 2

    # Create averaging kernel
    avg_kernel = torch.ones((1, 1, kernel_size, kernel_size),
                            device=depth.device,
                            dtype=depth.dtype) / (kernel_size**2)

    # Local mean: E[x]
    mean = F.conv2d(depth, avg_kernel, padding=pad)
    # Local mean of squares: E[x^2]
    mean_sq = F.conv2d(depth * depth, avg_kernel, padding=pad)

    # Variance: E[x^2] - (E[x])^2
    var = mean_sq - mean * mean

    # Mask of “low‐variance” regions
    mask = (var <= threshold).float()

    return mask


def mesh_get_depth(pts, color, extrins, focal_length, side_length, device):

    vertices, faces, colors = features_to_world_space_mesh(
        world_space_points=pts.detach(),
        colors=color.detach(),
        edge_threshold=0.48,
        H = side_length
    )

    mesh, o3d_device = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)

    inv_extrins = np.linalg.inv(extrins)

    depth, normals = mesh_rendering(mesh, focal_length, inv_extrins, o3d_device)

    return depth, normals


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_params)
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    if cfg.debugging:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
            mixed_precision=cfg.solver.mixed_precision,
            kwargs_handlers=[kwargs],
        )    

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
            mixed_precision=cfg.solver.mixed_precision,
            log_with="wandb",
            # project_dir="./mlruns",
            kwargs_handlers=[kwargs],
            # dispatch_batches = False,
        )

    # print(accelerator.use_distributed)
    # print(accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    from datetime import datetime
    now = datetime.now()
    formatted_now = now.strftime("%y%m%d_%H%M%S")
    
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}_{formatted_now}"
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    cfg.save_dir = save_dir
    
    # cfg save to yaml
    with open(f"{save_dir}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )

    # val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    
    if not cfg.inference:
        train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    if cfg.inference:
        val_scheduler = DDIMScheduler(**sched_kwargs)
        val_scheduler.set_timesteps(
            20, device="cuda")
        num_train_timesteps = val_scheduler.config.num_train_timesteps

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    # Option whether to start training from scratch or not
    train_from_scratch = cfg.train_from_scratch

    cond_channels = 16
    if cfg.use_depthmap:
        cond_channels += 1
    if cfg.use_conf:
        cond_channels += 1
    if cfg.use_normal:
        cond_channels += 3

    # Pose guider.
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320,
        conditioning_channels=cond_channels,
    ).to(device="cuda", dtype=weight_dtype)
    
    # TEST MARIGOLD
    
    if not train_from_scratch:
        # Reference Unet.
        reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                join(cfg.model_config_path, 'config.json')
        )).to(device="cuda", dtype=weight_dtype)        

        reference_unet.load_state_dict(torch.load(
            join(cfg.model_path, 'reference_unet.pth'),
            map_location= 'cpu'),
        )
        # Denoising Unet.
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            join(cfg.model_config_path, 'config.json'),
            join(cfg.model_path, 'denoising_unet.pth')
        ).to(device="cuda", dtype=weight_dtype)
            
        # Geometry Unets.
        
        path_dict = {
            "depth" : cfg.model_depth_path,
            "pointmap" : cfg.model_pointmap_path,
            "normal" : cfg.model_normal_path
        }

        if cfg.geo_first:
            if cfg.train_from_complete:
                try:
                    geometry_unet = UNet3DConditionModel.from_pretrained_2d(
                        join(cfg.model_config_path, 'geometry_config.json'),
                        join(path_dict[cfg.geo_setting], f'geometry_unet_{cfg.geo_setting}.pth')
                    ).to(device="cuda", dtype=weight_dtype)       
                except:
                    geometry_unet = UNet3DConditionModel.from_pretrained_2d(
                        join(path_dict[cfg.geo_setting], 'config.json'),
                        join(path_dict[cfg.geo_setting], 'diffusion_pytorch_model.bin')
                    ).to(device="cuda", dtype=weight_dtype)   
                    print("Loading directly from marigold")
            else:
                geometry_unet = UNet3DConditionModel.from_pretrained_2d(
                    join(path_dict[cfg.geo_setting], 'geometry_config.json'),
                    join(path_dict[cfg.geo_setting], 'geometry_unet.pth')
                ).to(device="cuda", dtype=weight_dtype)

        if cfg.geo_second:
            if cfg.train_from_complete:
                geometry_unet_2 = UNet3DConditionModel.from_pretrained_2d(
                    join(cfg.model_config_path, 'geometry_config.json'),
                    join(path_dict[cfg.geo_setting_2], f'geometry_unet_{cfg.geo_setting_2}.pth')
                ).to(device="cuda", dtype=weight_dtype)         
            else:
                geometry_unet_2 = UNet3DConditionModel.from_pretrained_2d(
                    join(path_dict[cfg.geo_setting_2], 'geometry_config.json'),
                    join(path_dict[cfg.geo_setting_2], 'geometry_unet.pth')
                ).to(device="cuda", dtype=weight_dtype)
        
        if cfg.geo_third:
            if cfg.train_from_complete:
                geometry_unet_3 = UNet3DConditionModel.from_pretrained_2d(
                    join(cfg.model_config_path, 'geometry_config.json'),
                    join(path_dict[cfg.geo_setting_3], f'geometry_unet_{cfg.geo_setting_3}.pth')
                ).to(device="cuda", dtype=weight_dtype)      

        if cfg.use_geo_ref_unet:
            geo_reference_unet = UNet2DConditionModel.from_config(
                UNet2DConditionModel.load_config(
                    join(cfg.model_config_path, 'geo_ref_config.json')
            )).to(device="cuda", dtype=weight_dtype)

            geo_reference_unet.load_state_dict(torch.load(
                # join(path_dict[cfg.geo_setting], 'geo_reference_unet-14000.pth'),
                join(path_dict[cfg.geo_setting], 'geo_reference_unet.pth'),
                map_location= 'cpu'),
            )        
                        
        if denoising_unet.conv_in.weight.shape[1] == 4:
            if cfg.use_warped_img_cond or cfg.switcher:
                inflated_in_weight = torch.cat([denoising_unet.conv_in.weight, denoising_unet.conv_in.weight], dim=1) / 2
                # inflated_out_weight = torch.cat([denoising_unet.conv_out.weight, denoising_unet.conv_out.weight], dim=0) / 2

                input_conv = InflatedConv3d(
                    8, 320, kernel_size=3, padding=(1, 1)
                ) 

                input_conv.weight = torch.nn.Parameter(inflated_in_weight) 
                # output_conv.weight = torch.nn.Parameter(inflated_out_weight)

                input_conv.bias = denoising_unet.conv_in.bias 
                # output_conv.bias = torch.nn.Parameter(torch.cat([denoising_unet.conv_out.bias,denoising_unet.conv_out.bias])) 

                denoising_unet.conv_in = input_conv
                # denoising_unet.conv_out = output_conv


        try:
            pose_guider.load_state_dict(torch.load(
                join(cfg.model_path, 'pose_guider.pth'),
                # "/workspace/data1/minseop/multiview-gen/checkpoints/co3d_naive/pose_guider.pth",
                map_location='cpu'),
            )
            print("LOADED!!")
        except:
            pass

    else:
        reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                join(cfg.model_path_scratch, 'config.json')
        )).to(device="cuda", dtype=weight_dtype)

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            join(cfg.model_path_scratch, 'config.json'),
            join(cfg.model_path_scratch, 'denoising_unet.bin')
        ).to(device="cuda", dtype=weight_dtype)

        ref_param_names = [ name for name,_ in reference_unet.named_parameters()]
        loading_params = {key : param for key, param in torch.load(join(cfg.model_path_scratch, 'denoising_unet.bin'),map_location= 'cpu').items()
                        if key in ref_param_names}
        reference_unet.load_state_dict(loading_params)
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    clip_preprocessor = CLIPImageProcessor()

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    # Explictly declare training models
    use_lora = cfg.use_lora
    
    #TODO : temporarily can be disabled for debugging
    pose_guider.requires_grad_(True)

    if use_lora == False:
        if not cfg.inference:
            if cfg.geo_first:
                geometry_unet.requires_grad_(True)
                for name, param in geometry_unet.named_parameters():
                    if "attn1.to_k" in name or "attn1.to_q" in name:
                        param.requires_grad_(False)
                                    
            if not cfg.train_geo_only:
                denoising_unet.requires_grad_(True)
                reference_unet.requires_grad_(True)

                for name, param in reference_unet.named_parameters():
                    if "up_blocks.3" in name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        
            else:
                denoising_unet.requires_grad_(False)
                reference_unet.requires_grad_(False)
                pose_guider.requires_grad_(False)
            
            if cfg.geo_second:
                geometry_unet_2.requires_grad_(True)
                for name, param in geometry_unet_2.named_parameters():
                    if "attn1.to_k" in name or "attn1.to_q" in name:
                        param.requires_grad_(False)
            
            if cfg.geo_third:
                geometry_unet_3.requires_grad_(True)
                for name, param in geometry_unet_3.named_parameters():
                    if "attn1.to_k" in name or "attn1.to_q" in name:
                        param.requires_grad_(False)
            
            if cfg.use_geo_ref_unet:
                geo_reference_unet.requires_grad_(True)
                for name, param in geo_reference_unet.named_parameters():
                    if "up_blocks.3" in name:
                        param.requires_grad_(False)
                
        else:
            denoising_unet.requires_grad_(False)
            reference_unet.requires_grad_(False)
            pose_guider.requires_grad_(False)            
            geometry_unet.requires_grad_(False)
            
            if cfg.geo_second:
                geometry_unet_2.requires_grad_(False)

            if cfg.geo_third:
                geometry_unet_3.requires_grad_(False)
            
            if cfg.use_geo_ref_unet:
                geo_reference_unet.requires_grad_(False)
            
    else:    
        denoising_unet.requires_grad_(False)
        reference_unet.requires_grad_(False)
        geometry_unet.requires_grad_(False)
        # less = [name for name, param in denoising_unet.named_parameters() if "transformer" in name]
   
        den_lora_params, den_train_names = inject_trainable_lora(denoising_unet)  
        ref_lora_params, ref_train_names = inject_trainable_lora(reference_unet)  

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
        feature_fusion_type=cfg.feature_fusion_type if hasattr(cfg, "feature_fusion_type") else "attention_masking",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        feature_fusion_type=cfg.feature_fusion_type if hasattr(cfg, "feature_fusion_type") else "attention_masking",
    )
    
    net_variables = [
        reference_unet,
        denoising_unet,
        None,
        None, # Second geometry network
        None, # Third geometry network
        None, # Geo reference unet
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    ]

    if cfg.geo_first:
        net_variables[2] = geometry_unet
    
    if cfg.geo_second:
        net_variables[3] = geometry_unet_2

    if cfg.geo_third:
        net_variables[4] = geometry_unet_3
    
    if cfg.use_geo_ref_unet:
        geo_reference_control_writer = ReferenceAttentionControl(
            geo_reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
            feature_fusion_type=cfg.feature_fusion_type if hasattr(cfg, "feature_fusion_type") else "attention_masking",
        )
        geo_reference_control_reader = ReferenceAttentionControl(
            geometry_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
            feature_fusion_type=cfg.feature_fusion_type if hasattr(cfg, "feature_fusion_type") else "attention_masking",
        )
        
        net_variables[5] = geo_reference_unet
        net_variables += [geo_reference_control_writer, geo_reference_control_reader]    
    
    net = Net(
        *net_variables
    )
    
    if cfg.inference:
        net.inference = True
    
    logger.info(f"Feature fusion type is '{cfg.feature_fusion_type}'")

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
            if cfg.geo_first:
                geometry_unet.enable_xformers_memory_efficient_attention()
            if cfg.geo_second:
                geometry_unet_2.enable_xformers_memory_efficient_attention()
            if cfg.geo_third:
                geometry_unet_3.enable_xformers_memory_efficient_attention()
            if cfg.use_geo_ref_unet:
                geo_reference_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        geometry_unet.enable_gradient_checkpointing()
        if cfg.geo_second:
            geometry_unet_2.enable_gradient_checkpointing()
        if cfg.geo_third:
            geometry_unet_3.enable_gradient_checkpointing()
        if cfg.use_geo_ref_unet:
            geo_reference_unet.enable_gradient_checkpointing()
    
    if not cfg.inference:
        if cfg.solver.scale_lr:
            learning_rate = (
                cfg.solver.learning_rate
                * cfg.solver.gradient_accumulation_steps
                * cfg.dataset.train_bs
                * accelerator.num_processes
            )
        else:
            learning_rate = cfg.solver.learning_rate

        # Initialize the optimizer
        if cfg.solver.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))

        optimizer = optimizer_cls(
            trainable_params,
            lr=learning_rate,
            betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
            weight_decay=cfg.solver.adam_weight_decay,
            eps=cfg.solver.adam_epsilon,
        )
        
        optimizer_load = cfg.optimizer_load
        
        if optimizer_load:
            try:
                optimizer.load_state_dict(
                    torch.load(join(cfg.model_path, 'optimizer.pth'),
                        map_location='cpu'),
                )
                print("Optimizer loaded!!!!!!!!!!")
            except:
                pass
            
        # Scheduler #
        lr_scheduler = get_scheduler(
            cfg.solver.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.solver.lr_warmup_steps
            * cfg.solver.gradient_accumulation_steps,
            num_training_steps=cfg.solver.max_train_steps
            * cfg.solver.gradient_accumulation_steps,
        )
        
        if optimizer_load:
            try:
                scheduler_state = torch.load(join(cfg.model_path, "scheduler.bin")),
                lr_scheduler.load_state_dict(scheduler_state[0])
            except:
                pass
    
    else:
        pass

    dataset = cfg.dataset.name
    num_viewpoints = cfg.dataset.num_viewpoints
    num_ref_viewpoints = cfg.dataset.num_ref

    ENDPOINT_URL = 'https://storage.clova.ai'

    os.environ['AWS_ACCESS_KEY_ID'] = "AUIVA2ODFS9S2YDD0A75"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "VDIVVqIC9FCC0GmOQ2nNy3o7NjkWVqC4oTDOz3mM"
    os.environ['S3_ENDPOINT_URL'] = ENDPOINT_URL

    num_list = torch.arange(0, 1200)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    # Dataloader sections
    co3d_directory = cfg.co3d_directory
    realestate_directory = cfg.realestate_directory      

    urls_1 = []
    for root, _, files in os.walk(realestate_directory):
        for file in files:
            if file.endswith(".tar"):
                urls_1.append(os.path.join(root, file))
    
    urls_2 = []
    for root, _, files in os.walk(co3d_directory):
        for file in files:
            if file.endswith(".tar"):
                urls_2.append(os.path.join(root, file))
    
    if cfg.realestate_only:
        urls = urls_1
    else:
        urls = urls_1 + urls_2
    
    shardshuffle=True
    resampled=True
    dataset_length = 2000
    epoch = 10000


    if not cfg.train_vggt:
        postprocess_fn_1 = partial(postprocess_combined, num_viewpoints=num_ref_viewpoints, interpolate_only = cfg.interpolate_only)
    else:
        postprocess_fn_1 = partial(postprocess_vggt, num_viewpoints=num_ref_viewpoints, interpolate_only = cfg.interpolate_only, view_range=cfg.view_range)

    train_dataset = (
            wds.WebDataset(urls, 
                            resampled=resampled,
                            shardshuffle=shardshuffle, 
                            nodesplitter=wds.split_by_node,
                            workersplitter=wds.split_by_worker,
                            handler=wds.ignore_and_continue)
            .decode("pil")
            .map(postprocess_fn_1)
            .with_length(dataset_length)
            .with_epoch(cfg.dataset.train_bs * epoch)
    )

    train_dataloader = DataLoader(train_dataset, num_workers=world_size, batch_size=cfg.dataset.train_bs, persistent_workers=True)
    
    if cfg.inference:
        (
            net,
            train_dataloader,
        ) = accelerator.prepare(
            net,
            train_dataloader,
        )
        
    else:
        (
            net,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            net,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            project_name="animate-nvs",
            config=OmegaConf.to_container(cfg),
        )

    # import pdb; pdb.set_trace()

    # Train!
    total_batch_size = (
        cfg.dataset.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dataset.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            raise ValueError("Do not support latest checkpoint currently")
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    embedder, out_dim = get_embedder(2)
    
    if cfg.geo_first:
        if cfg.train_geo_only:
            attn_proc_hooker=XformersCrossAttentionHooker(False, num_ref_views=num_ref_viewpoints, setting="writing", cross_attn=cfg.use_geo_ref_unet)
        else:
            attn_proc_hooker=XformersCrossAttentionHooker( True, num_ref_views=num_ref_viewpoints, setting="writing", cross_attn=cfg.use_geo_ref_unet)
            
        denoising_unet.set_attn_processor(attn_proc_hooker)
        geo_attn_proc_hooker=XformersCrossAttentionHooker( True, num_ref_views=num_ref_viewpoints, setting="reading", cross_attn=cfg.use_geo_ref_unet)
        geometry_unet.set_attn_processor(geo_attn_proc_hooker)
    
    else:
        attn_proc_hooker = None
        geo_attn_proc_hooker = None
    
    if cfg.geo_second:
        geo_attn_proc_hooker_2=XformersCrossAttentionHooker( True, num_ref_views=num_ref_viewpoints, setting="reading", cross_attn=cfg.use_geo_2_ref_unet)
        geometry_unet_2.set_attn_processor(geo_attn_proc_hooker_2)

    if cfg.geo_third:
        geo_attn_proc_hooker_3=XformersCrossAttentionHooker( True, num_ref_views=num_ref_viewpoints, setting="reading", cross_attn=False)
        geometry_unet_3.set_attn_processor(geo_attn_proc_hooker_3)
    
    # Pointmap embedder initialization
    if cfg.embed_pointmap_norm:        
        if cfg.train_vggt:
            ptsmap_min = torch.tensor([-0.6338, -0.4921, 0.4827]).to(image_enc.device)
            ptsmap_max = torch.tensor([ 0.6190, 0.6307, 1.6461]).to(image_enc.device)            
        else:   
            ptsmap_min = torch.tensor([-0.1798, -0.2254,  0.0593]).to(image_enc.device)
            ptsmap_max = torch.tensor([0.1899, 0.0836, 0.7480]).to(image_enc.device)
            
        pts_norm_func = PointmapNormalizer(ptsmap_min, ptsmap_max, k=0.9)

    psnr_list =[]
    
    for epoch in range(first_epoch, num_train_epochs):

        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            if cfg.inference:
                if not cfg.inference_from_data:
                    fin_frame = 2
                    scene_num = step // 3
                    frame_num = step % 3

            with accelerator.accumulate(net):
                # Convert videos to latent space 
                if not cfg.inference_from_data:
                    # if cfg.dataset.num_ref != 1: 
                    if not cfg.interpolate_only:                          
                        target_idx = random.sample(range(cfg.dataset.num_viewpoints),1)[0]
                    else:
                        target_idx = random.sample(range(1,cfg.dataset.num_viewpoints-1),1)[0]
                
                    src_idx = [k for k in range(num_viewpoints) if k is not target_idx]
                
                else:
                    target_idx = 0
                    src_idx = [k for k in range(num_viewpoints) if k is not target_idx]

                images = dict(ref=[batch["image"][:,k].unsqueeze(1) for k in src_idx], tgt=batch["image"][:,target_idx])

                with torch.no_grad():
                    # Dataloader
                    device = image_enc.device

                    if not cfg.train_vggt:
                        batch_size = batch['pose'].shape[0]

                        ref_camera=dict(pose=batch['pose'][:,src_idx].float(), 
                            focals=batch['focals'][:,src_idx], 
                            orig_img_size=torch.tensor([512, 512]).to(device))

                        tgt_camera=dict(pose=batch['pose'][:,target_idx].float(),
                            focals=batch['focals'][:, target_idx],
                            orig_img_size=torch.tensor([512, 512]).to(device))
                    
                        closest_idx = find_closest_camera(reference_cameras=ref_camera["pose"], target_pose=tgt_camera["pose"])
                        camera_info = dict(ref=ref_camera, tgt=tgt_camera)
                        correspondence = dict(ref=batch['points'][:, src_idx].float(), tgt=batch['points'][:, target_idx].float())
                        
                    else:
                        batch_size, num_view, _, _ = batch['extrinsic'].shape
                        pointmaps = batch['points'].permute(0,1,3,4,2)
                        w2c = torch.cat((batch['extrinsic'], torch.tensor([0,0,0,1]).to(device)[None,None,None,...].repeat(batch_size, num_view, 1, 1)), dim=-2)
                        extrinsic = torch.linalg.inv(w2c)
                                                    
                        if cfg.normalized_pose:
                            extrinsic = torch.matmul(w2c[:,target_idx].unsqueeze(1), extrinsic)
                            pointmaps = torch.matmul(w2c[:,target_idx][:,None,None,None,:3,:3], pointmaps[...,None]).squeeze(-1) + w2c[:,target_idx][:,None,None,None,:3,3]
                                                    
                        focal_list = []
                        
                        for i in batch["intrinsic"]:
                            min_val, idx = i[0,:2,2].min(dim=-1)
                            focal_list.append(i[:,int(idx),int(idx)] * (256 / min_val))
                        
                        focal = torch.stack(focal_list)[...,None]
                                                        
                        ref_camera=dict(pose=extrinsic[:,src_idx].float(), 
                            focals=focal[:,src_idx], 
                            orig_img_size=torch.tensor([512, 512]).to(device))

                        tgt_camera=dict(pose=extrinsic[:,target_idx].float(),
                            focals=focal[:,target_idx],
                            orig_img_size=torch.tensor([512, 512]).to(device))
                    
                        closest_idx = find_closest_camera(reference_cameras=ref_camera["pose"], target_pose=tgt_camera["pose"])
                        camera_info = dict(ref=ref_camera, tgt=tgt_camera)
                        correspondence = dict(ref=pointmaps[:, src_idx].float(), tgt=pointmaps[:, target_idx].float())
                                                    
                        if batch['dataset'][0][0] == 1:
                            current_dataset = "realestate"
                        else:
                            current_dataset = "co3d"
                                                        
                    # full conditioning ratio
                    if cfg.random_full_condition:
                        full_cond_variable = torch.rand(1)
                        
                        if full_cond_variable < cfg.full_cond_ratio:
                            full_condition = True
                        else:
                            full_condition = False
                        
                        # if step % 3 == 0:
                        #     full_condition = True
                        # else:
                        #     full_condition = False
                            
                    else:
                        full_condition = False
                                                                
                    if current_dataset == "realestate" and cfg.nomesh_normal:
                        mesh_normals = convert_depth_to_normal(correspondence["tgt"].permute(0,3,1,2)[:,2:])
                        
                        tgt_origins = tgt_camera['pose'][:,:3,-1]
                        ref_dist = torch.linalg.norm(correspondence["ref"] - tgt_origins[...,None,None,None,:],axis=-1)[...,None]
                        mesh_ref_normals = convert_depth_to_normal(ref_dist.permute(0,1,4,2,3).reshape(batch_size * num_ref_viewpoints,1,512,512)).reshape(batch_size, num_ref_viewpoints, 3, 512, 512)
                                                
                        norm_depth = ref_dist
                                                
                    else:
                        if cfg.use_mesh or cfg.use_normal:
                            tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth = mari_embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, batch_size, device, full_condition=full_condition)
                    
                    src_images = images["ref"]
                                                            
                    args = dict(
                        src_images=src_images,
                        correspondence=correspondence,
                        camera_info=camera_info,
                        embedder=embedder,
                        src_idx=src_idx,
                        tgt_idx=target_idx,
                        dataset=dataset,
                        tgt_image=images["tgt"],
                        point_masking=cfg.point_masking,
                        masking_percent=cfg.masking_percent,
                        use_gt_cor=cfg.use_gt_cor,
                        pure_gt=cfg.pure_gt,
                        gt_threshold=cfg.gt_mask_threshold,
                        current_dataset=current_dataset,
                        full_condition=full_condition,
                        nomesh_normal=cfg.nomesh_normal
                    )

                    if cfg.use_mesh:
                        args["mesh_pts"] = mesh_pts
                        args["mesh_depth"] = mesh_depth
                    
                    if cfg.use_normal:
                        args["mesh_normals"] = mesh_normals.to(device)
                        args["mesh_ref_normals"] = mesh_ref_normals.to(device)

                    if cfg.use_depthmap:
                        args["ref_depth"] = norm_depth

                    if cfg.use_conf:
                        args["confidence"] = confidence_map

                    if cfg.gt_cor_reg:
                        args["gt_cor_regularize"] = True
                        
                    if cfg.embed_pointmap_norm:
                        args["pts_norm_func"] = pts_norm_func
                        
                    if cfg.use_normal_mask:
                        args["normal_mask"] = mesh_normal_mask.to(device)
                    
                    conditions, renders = prepare_duster_embedding(**args)
                    latents = vae.encode(images["tgt"].to(weight_dtype) * 2 - 1).latent_dist.sample()

                    if cfg.geo_second:
                        geo_setting_2 = cfg.geo_setting_2 
                    else:
                        geo_setting_2 = None
                
                    if cfg.geo_third:
                        geo_setting_3 = cfg.geo_setting_3
                    else:
                        geo_setting_3 = None
                    
                    if cfg.geo_setting == "depth" or geo_setting_2 == "depth":                          
                        if geo_setting_2 == "depth":
                            geo_latents_2 = vae.encode(tgt_depth_norm.to(weight_dtype)).latent_dist.sample()
                        else:
                            geo_latents = vae.encode(tgt_depth_norm.to(weight_dtype)).latent_dist.sample()
                        
                        if cfg.use_warped_geo_cond_at_geo:
                            warped_depth = mesh_depth.unsqueeze(1).repeat(1,3,1,1)
                            norm_warped_depth = depth_normalize(cfg, warped_depth)
                            norm_warped_depth = torch.clip(norm_warped_depth, min=-1.0, max=1.0)
                            warped_geo_encoded = vae.encode(norm_warped_depth).latent_dist.sample()

                    
                    if cfg.geo_setting == "pointmap" or geo_setting_2 == "pointmap":
                        ref_pointmaps = correspondence["ref"].reshape(-1,512,512,3).permute(0,3,1,2)
                        tgt_pointmap = correspondence["tgt"].permute(0,3,1,2)

                        if cfg.use_warped_geo_cond_at_geo:   
                            warped_tgt_pointmap = mesh_pts.permute(0,3,1,2)
                        
                        if cfg.conditioning_pointmap_norm:
                            minmax_set = True if current_dataset != "realestate" else False
                            
                            if minmax_set:
                                if cfg.train_vggt:
                                    ptsmap_min = torch.tensor([-0.6338, -0.4921, 0.4827]).to(image_enc.device)
                                    ptsmap_max = torch.tensor([ 0.6190, 0.6307, 1.6461]).to(image_enc.device)            
                                else:   
                                    ptsmap_min = torch.tensor([-0.1798, -0.2254,  0.0593]).to(image_enc.device)
                                    ptsmap_max = torch.tensor([0.1899, 0.0836, 0.7480]).to(image_enc.device)
                                
                                ref_pointmaps = torch.clip((ref_pointmaps - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                                tgt_pointmap = torch.clip((tgt_pointmap - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)

                            else:
                                ptsmap_min = correspondence["ref"].reshape(batch_size,-1,3).min(dim=-2)[0]
                                ptsmap_max = correspondence["ref"].reshape(batch_size,-1,3).max(dim=-2)[0]

                                ref_ptsmap_min = ptsmap_min.unsqueeze(1).repeat(1,num_ref_viewpoints,1).reshape(-1,3)
                                ref_ptsmap_max = ptsmap_max.unsqueeze(1).repeat(1,num_ref_viewpoints,1).reshape(-1,3)   
                                                                
                                ref_pointmaps = torch.clip((ref_pointmaps - ref_ptsmap_min[...,None,None]) / (ref_ptsmap_max[...,None,None] - ref_ptsmap_min[...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                                tgt_pointmap = torch.clip((tgt_pointmap - ptsmap_min[...,None,None]) / (ptsmap_max[...,None,None] - ptsmap_min[...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                                          
                        if cfg.use_warped_geo_cond_at_geo:   
                            warped_tgt_pointmap = torch.clip((warped_tgt_pointmap - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                            warped_geo_encoded = vae.encode(warped_tgt_pointmap.to(weight_dtype)).latent_dist.sample()
                        
                        if geo_setting_2 == "pointmap":
                            geo_latents_2 = vae.encode(tgt_pointmap.to(weight_dtype)).latent_dist.sample()
                        else:
                            geo_latents = vae.encode(tgt_pointmap.to(weight_dtype)).latent_dist.sample()
                    
                    if cfg.geo_setting == "normal" or geo_setting_2 == "normal" or geo_setting_3 == "normal":
                        normal = convert_depth_to_normal(tgt_depth_norm[:,:1])
                        
                        if geo_setting_3 == "normal": 
                            geo_latents_3 = vae.encode(normal.to(weight_dtype)).latent_dist.sample()
                        elif geo_setting_2 == "normal":
                            geo_latents_2 = vae.encode(normal.to(weight_dtype)).latent_dist.sample()
                        else:
                            geo_latents = vae.encode(normal.to(weight_dtype)).latent_dist.sample()

                    if cfg.use_warped_geo_cond_at_geo:
                        warped_geo_encoded = warped_geo_encoded.unsqueeze(2) 
                        warped_geo_encoded = warped_geo_encoded * 0.18215  
                                            
                    geo_latents = geo_latents.unsqueeze(2) 
                    geo_latents = geo_latents * 0.18215  
                    
                    if cfg.geo_second:
                        geo_latents_2 = geo_latents_2.unsqueeze(2) 
                        geo_latents_2 = geo_latents_2 * 0.18215  

                    if cfg.geo_third:
                        geo_latents_3 = geo_latents_3.unsqueeze(2) 
                        geo_latents_3 = geo_latents_3 * 0.18215  
                    
                    # import pdb; pdb.set_trace()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215    

                    if cfg.ref_net_expand:
                        ref_input = conditions["ref_correspondence"]
                        ref_encoded_batches = []
                        for ref_corrs in ref_input:
                            ref_encoded_batches.append(vae.encode(ref_corrs.permute(0,3,1,2)).latent_dist.sample()[None,...])
                        ref_corr_latents = torch.cat(ref_encoded_batches, dim=0).permute(1,0,2,3,4)
                        
                    # if cfg.use_warped_img_cond:
                    zero_mask = (renders["warped"][:,:1] != 0.).float()
                    warped_image = zero_mask * (renders["warped"] * 2 - 1)
                    if cfg.use_normal_mask and current_dataset != "realestate":
                        warped_image = warped_image * mesh_normal_mask.permute(0,3,1,2)  
                    warped_latents = vae.encode(warped_image).latent_dist.sample().unsqueeze(2)  
                    warped_latents = warped_latents *  0.18215    
                                                            
                is_warped_feat_injection = cfg.feature_fusion_type == 'warped_feature'
                   
                noise = torch.randn_like(latents)

                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                
                # Sample a random timestep for each video
                if not cfg.inference:
                    timesteps = torch.randint(
                        0,
                        train_noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                else:
                    # when inferencing, timestep.
                    timesteps = val_scheduler.timesteps
                                    
                timesteps = timesteps.long()

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_depth_list = []
                ref_image_list = []

                ref_stack = torch.cat(images["ref"], dim=1)
                B, V, C, H, W = ref_stack.shape

                ref_stack = ref_stack.reshape(-1,C,H,W) * 2 - 1 # Normalization
                
                if cfg.use_geo_ref_unet:
                    for batch_idx, (ref_d, ref_img, clip_img) in enumerate(
                        zip(
                            ref_depth if cfg.geo_setting == "depth" else ref_pointmaps,
                            ref_stack,
                            clip_preprocessor(ref_stack*0.5+0.5, do_rescale=False, return_tensors="pt").pixel_values,
                        )
                    ):  
                        if uncond_fwd:
                            clip_image_list.append(torch.zeros_like(clip_img))
                        else:
                            clip_image_list.append(clip_img)
                            
                        ref_depth_list.append(ref_d)
                        ref_image_list.append(ref_img)
                else:
                    for batch_idx, (ref_img, clip_img) in enumerate(
                        zip(
                            ref_stack,
                            clip_preprocessor(ref_stack*0.5+0.5, do_rescale=False, return_tensors="pt").pixel_values,
                        )
                    ):  
                        if uncond_fwd:
                            clip_image_list.append(torch.zeros_like(clip_img))
                        else:
                            clip_image_list.append(clip_img)
                            
                        ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img_stack = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img_stack
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215
                    
                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                    
                    if cfg.use_geo_ref_unet:
                        ref_depth_stack = torch.stack(ref_depth_list, dim=0).to(
                            dtype=vae.dtype, device=vae.device
                        )
                        ref_depth_latents = vae.encode(
                            ref_depth_stack
                        ).latent_dist.sample()  # (bs, d, 64, 64)
                        
                        ref_depth_latents = ref_depth_latents * 0.18215
                
                # add noise
                if not cfg.inference:
                    noisy_latents = train_noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    
                    geo_noisy_latents = train_noise_scheduler.add_noise(
                        geo_latents, noise, timesteps
                    )
                    
                    if cfg.geo_second:
                        geo_noisy_latents_2 = train_noise_scheduler.add_noise(
                            geo_latents_2, noise, timesteps
                        )

                    if cfg.geo_third:
                        geo_noisy_latents_3 = train_noise_scheduler.add_noise(
                            geo_latents_3, noise, timesteps
                        )
                
                    # Get the target for loss depending on the prediction type
                    if train_noise_scheduler.prediction_type == "epsilon":
                        target = noise
                        geo_target = noise
                        
                        if cfg.geo_second:
                            geo_target_2 = noise

                        if cfg.geo_third:
                            geo_target_3 = noise
                            
                        
                    elif train_noise_scheduler.prediction_type == "v_prediction":
                        target = train_noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                        geo_target = train_noise_scheduler.get_velocity(
                            geo_latents, noise, timesteps
                        )
                        
                        if cfg.geo_second:
                            geo_target_2 = train_noise_scheduler.get_velocity(
                                geo_latents_2, noise, timesteps
                            )

                        if cfg.geo_third:
                            geo_target_3 = train_noise_scheduler.get_velocity(
                                geo_latents_3, noise, timesteps
                            )
                        
                    else:
                        raise ValueError(
                            f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                        )
                
                else:                        
                    initial_t = torch.tensor(
                        [999] * batch_size
                    ).to(device, dtype=torch.long)
                        
                    latents_noisy_start = val_scheduler.add_noise(
                        latents, noise, initial_t
                    )
                    
                    noisy_latents = latents_noisy_start
                    geo_noisy_latents = latents_noisy_start
                    if cfg.geo_second:
                        geo_noisy_latents_2 = latents_noisy_start
                    if cfg.geo_third:
                        geo_noisy_latents_3 = latents_noisy_start
                

                # Image embeddings ------------------
                image_prompt_embeds = image_prompt_embeds.reshape(B,V,1,-1).permute(1,0,2,3)
                ref_image_latents = ref_image_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
                task_emb = None

                if cfg.use_warped_img_cond:
                    noisy_latents = torch.cat((noisy_latents, warped_latents), dim=1)
                
                # Geometry embeddings ------------ (always have warped latents as condition)
                geo_noisy_latents = torch.cat((warped_latents, geo_noisy_latents), dim=1)
                
                # Partial geometry condition to geometry embedding
                if cfg.use_warped_geo_cond_at_geo:
                    geo_noisy_latents = torch.cat((geo_noisy_latents, warped_geo_encoded), dim=1)
                
                # Reference network for geometry
                if cfg.use_geo_ref_unet:
                    ref_depth_latents = ref_depth_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
                    geo_ref_latents = torch.cat((ref_image_latents, ref_depth_latents), dim=2).to(weight_dtype)
                else:
                    geo_ref_latents = None
                
                # Second geometry network -----------------
                if cfg.geo_second:
                    geo_noisy_latents_2 = torch.cat((warped_latents, geo_noisy_latents_2), dim=1).to(weight_dtype)
                else:
                    geo_noisy_latents_2 = None

                # Third geometry network -----------------
                if cfg.geo_third:
                    geo_noisy_latents_3 = torch.cat((warped_latents, geo_noisy_latents_3), dim=1).to(weight_dtype)
                else:
                    geo_noisy_latents_3 = None
                                
                if not cfg.inference:
                    results_dict = net(
                        noisy_latents.to(weight_dtype),
                        geo_noisy_latents.to(weight_dtype),
                        geo_noisy_latents_2,
                        geo_noisy_latents_3,
                        timesteps,
                        ref_image_latents.to(weight_dtype),
                        geo_ref_latents,
                        image_prompt_embeds.to(weight_dtype),
                        conditions["ref_embeds"], # V List of (B, 16, 512, 512) latents
                        conditions["tgt_embed"].to(weight_dtype), # Tensor of (B, 16, 512, 512)
                        correspondence=None,
                        weight_dtype=weight_dtype,
                        gt_target_coord_embed = conditions['gt_tgt_embed'] if conditions['gt_tgt_embed'] != None else None,
                        task_emb=task_emb,
                        attn_proc_hooker=attn_proc_hooker,
                        geo_attn_proc_hooker=geo_attn_proc_hooker if cfg.geo_first else None,
                        geo_attn_proc_hooker_2=geo_attn_proc_hooker_2 if cfg.geo_second else None,
                        geo_attn_proc_hooker_3=geo_attn_proc_hooker_3 if cfg.geo_third else None,
                        closest_idx = closest_idx
                    )
                                    
                else:
                    with torch.no_grad():
                        results_dict = net(
                            noisy_latents.to(weight_dtype),
                            geo_noisy_latents.to(weight_dtype),
                            geo_noisy_latents_2,
                            geo_noisy_latents_3,
                            timesteps,
                            ref_image_latents.to(weight_dtype),
                            geo_ref_latents,
                            image_prompt_embeds.to(weight_dtype),
                            conditions["ref_embeds"], # V List of (B, 16, 512, 512) latents
                            conditions["tgt_embed"].to(weight_dtype), # Tensor of (B, 16, 512, 512)
                            correspondence=None,
                            weight_dtype=weight_dtype,
                            gt_target_coord_embed = conditions['gt_tgt_embed'] if conditions['gt_tgt_embed'] != None else None,
                            task_emb=task_emb,
                            attn_proc_hooker=attn_proc_hooker,
                            geo_attn_proc_hooker=geo_attn_proc_hooker,
                            geo_attn_proc_hooker_2=geo_attn_proc_hooker_2 if cfg.geo_second else None,
                            geo_attn_proc_hooker_3=geo_attn_proc_hooker_3 if cfg.geo_third else None,
                            val_scheduler=val_scheduler,
                            vae = vae
                        )
                        
                        sys_random = random.SystemRandom()
                        num = sys_random.randint(0,10000)
                        
                        diff_gt_ori = torch.sqrt((images["tgt"] - results_dict['img_pred'])**2).sum(dim=1).unsqueeze(1)
                        diff_gt = apply_heatmap(diff_gt_ori).to(device)

                        downsampled_gt = F.interpolate(images["tgt"], size=(256, 256), mode='bilinear', align_corners=False)
                        downsampled_pred = F.interpolate(results_dict['img_pred'], size=(256, 256), mode='bilinear', align_corners=False)

                        mse = F.mse_loss(images["tgt"], results_dict['img_pred'], reduction='mean')
                        down_mse = F.mse_loss(downsampled_gt, downsampled_pred, reduction='mean')
                        psnr = 10 * math.log10(1.0 / (down_mse.item()+1e-9))
                        
                        ssim_value = ssim(downsampled_gt, downsampled_pred)

                        space = torch.ones(1,3,512,80).to(device)
                        
                        try:
                            depth_map = torch.mean(results_dict['geo_pred_2'], dim=1, keepdim=True).repeat(1,3,1,1)                   
                            stack_img = torch.cat((images["tgt"].to(device), space, warped_image * 0.5 + 0.5, space, tgt_pointmap*0.5+0.5, space, tgt_depth_norm*0.5+0.5, space, results_dict['img_pred'], space, results_dict['geo_pred'][:,2:].repeat(1,3,1,1), space, depth_map, space, diff_gt), dim=-1).detach()[0]  
                        except:
                            stack_img = torch.cat((images["tgt"].to(device), space, warped_image * 0.5 + 0.5, space, tgt_pointmap*0.5+0.5, space, tgt_depth_norm*0.5+0.5, space, results_dict['img_pred'], space, results_dict['geo_pred'][:,2:].repeat(1,3,1,1)), dim=-1).detach()[0]  
                        
                        gen_img = torch.cat((images["tgt"].to(device), space, results_dict['img_pred'], space, diff_gt), dim=-1).detach()[0]

                        if not cfg.inference_from_data:
                        
                            if frame_num == 0:
                                ult_stack = []
                                img_stack = []
                                psnrs = []
                                ult_stack.append(stack_img)
                                img_stack.append(gen_img)
                                psnrs.append(psnr)
                        
                            elif frame_num != fin_frame:
                                ult_stack.append(stack_img)
                                img_stack.append(gen_img)
                                psnrs.append(psnr)
                            
                            elif frame_num == fin_frame:
                                ult_stack.append(stack_img)
                                img_stack.append(gen_img)
                                psnrs.append(psnr)

                                now = strftime("%m_%d_%H_%M_%S", gmtime())
                                make_video(ult_stack, now, folder_name="everything")
                                make_video(img_stack, now, folder_name="images_only")

                                ref_img = torch.cat(images["ref"]).squeeze()
                                image_dir = os.path.join("outputs", f"{now}/{exp_name}_reference_frames.png")
                                save_image(ref_img, image_dir)

                                print(psnrs)
                                avg_psnr = sum(psnrs) / len(psnrs)
                                print(f"Average PSNR value: {avg_psnr}")
                        
                        else:
                            print(psnr)
                            psnr_list.append(psnr)
                            now = strftime("%m_%d_%H_%M_%S", gmtime())
                            save_image(stack_img, f"val_{now}.png")

                            if step == 30:
                                average = sum(psnr_list) / len(psnr_list)
                                print(f"PSNR: {average}")

                                raise ValueError
                                                                        
                        continue
                                    
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1

                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                                        
                    img_loss = F.mse_loss(
                        results_dict["img_pred"].float(), target.float(), reduction="none"
                    )
                                                
                    img_loss = (
                        img_loss.mean(dim=list(range(1, len(img_loss.shape))))
                        * mse_loss_weights
                    )
                    
                    if cfg.geo_first:
                        
                        geo_pred = results_dict["geo_pred"].float()
                        
                        geo_loss = F.mse_loss(
                           geo_pred, geo_target.float(), reduction="none"
                        )
                        
                        if cfg.inpaint_emph:
                            with torch.no_grad():
                                mask = (warped_image[:,:1] == 0).float()
                                inpaint_mask = F.interpolate(mask, size=(64,64), mode="nearest")
                                ones = torch.ones_like(inpaint_mask)
                                
                                emph_mask = (ones + 3 * inpaint_mask).to(device).unsqueeze(2)
                            
                            geo_loss = geo_loss * emph_mask                            

                        geo_loss = (
                            geo_loss.mean(dim=list(range(1, len(geo_loss.shape))))
                            * mse_loss_weights
                        )
                        
                        geo_loss = geo_loss.mean()
                    
                    else:
                        geo_loss = 0.
                    
                    if cfg.geo_second:                                            
                        geo_loss_2 = F.mse_loss(
                            results_dict["geo_pred_2"].float(), geo_target_2.float(), reduction="none"
                        )
                        
                        geo_loss_2 = (
                            geo_loss_2.mean(dim=list(range(1, len(geo_loss_2.shape))))
                            * mse_loss_weights
                        )
                        
                        geo_loss_2 = geo_loss_2.mean()
                    
                    else:
                        geo_loss_2 = 0.

                    if cfg.geo_third:                                            
                        geo_loss_3 = F.mse_loss(
                            results_dict["geo_pred_3"].float(), geo_target_3.float(), reduction="none"
                        )
                        
                        geo_loss_3 = (
                            geo_loss_3.mean(dim=list(range(1, len(geo_loss_3.shape))))
                            * mse_loss_weights
                        )
                        
                        geo_loss_3 = geo_loss_3.mean()
                    
                    else:
                        geo_loss_3 = 0.

                img_loss = img_loss.mean()
                            
                if not cfg.train_geo_only:
                    loss = cfg.img_weight * img_loss + cfg.weight_1 * geo_loss + cfg.weight_2 * geo_loss_2 + cfg.weight_3 * geo_loss_3
 
                else:
                    loss = geo_loss + geo_loss_2 + geo_loss_3
                                                        
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.dataset.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )

                # import pdb; pdb.set_trace()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
 
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                if cfg.geo_first:
                    attn_proc_hooker.clear()
                    geo_attn_proc_hooker.clear()
                if cfg.geo_second:
                    geo_attn_proc_hooker_2.clear()
                if cfg.geo_third:
                    geo_attn_proc_hooker_3.clear()
                if cfg.use_geo_ref_unet:
                    geo_reference_control_reader.clear()
                    geo_reference_control_writer.clear()
                    
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"img_loss": img_loss}, step=global_step)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if cfg.geo_first:
                    accelerator.log({"geo_loss": geo_loss}, step=global_step)
                if cfg.geo_second:
                    accelerator.log({"geo_loss_2": geo_loss_2}, step=global_step)
                if cfg.geo_third:
                    accelerator.log({"geo_loss_3": geo_loss_3}, step=global_step)
                if cfg.random_full_condition:
                    if full_condition:
                        accelerator.log({"full_cond_loss": train_loss}, step=global_step)
                        accelerator.log({"partial_loss": 0.0}, step=global_step)
                    else:
                        accelerator.log({"full_cond_loss": 0.0}, step=global_step)
                        accelerator.log({"partial_loss": train_loss}, step=global_step)
                
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)
                        unwrap_net = accelerator.unwrap_model(net)
                        save_checkpoint(
                            unwrap_net.reference_unet,
                            save_dir,
                            "reference_unet",
                            global_step,
                            total_limit=1,
                        )
                        save_checkpoint(
                            unwrap_net.denoising_unet,
                            save_dir,
                            "denoising_unet",
                            global_step,
                            total_limit=1,
                        )
                        save_checkpoint(
                            unwrap_net.pose_guider,
                            save_dir,
                            "pose_guider",
                            global_step,
                            total_limit=1,
                        )
                        save_checkpoint(
                            optimizer,
                            save_dir,
                            "optimizer",
                            global_step,
                            total_limit=1
                        )
                        
                        if cfg.geo_first:
                            save_checkpoint(
                                unwrap_net.geometry_unet,
                                save_dir,
                                "geometry_unet",
                                global_step,
                                total_limit=1,
                            )

                        if cfg.use_geo_ref_unet:
                            save_checkpoint(
                                unwrap_net.geo_reference_unet,
                                save_dir,
                                "geo_reference_unet",
                                global_step,
                                total_limit=1
                            )
                        
                        if cfg.geo_second:
                            save_checkpoint(
                                unwrap_net.geometry_unet_2,
                                save_dir,
                                "geometry_unet_2",
                                global_step,
                                total_limit=1,
                            )

                        if cfg.geo_third:
                            save_checkpoint(
                                unwrap_net.geometry_unet_3,
                                save_dir,
                                "geometry_unet_3",
                                global_step,
                                total_limit=1,
                            )
                        

                # if global_step % cfg.val.validation_steps == 0:
                # with torch.no_grad():
                #     psnr_mean, imgs = log_validation(net, accelerator, ref_reader=reference_control_reader, ref_writer=reference_control_writer, multitask = cfg.multitask, depth_condition=cfg.use_depthmap, use_mesh=cfg.use_mesh, use_normal=cfg.use_normal)
                #     accelerator.log({"psnr": psnr_mean}, step=global_step)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            del loss
            del results_dict
            
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (epoch + 1) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=1,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=1,
            )
            save_checkpoint(
                unwrap_net.pose_guider,
                save_dir,
                "pose_guider",
                global_step,
                total_limit=1,
            )
            save_checkpoint(
                optimizer,
                save_dir,
                "optimizer",
                global_step,
                total_limit=1
            )

        # if global_step % cfg.val.validation_steps == 0:


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def load_16bit_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
        
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=Image.BILINEAR),  # Resize shorter side to 512
            transforms.CenterCrop(512),  # Center crop to 512x512
        ])
        
        depth = transform(torch.tensor(depth[None,...]))
        
    return depth

# def load_16bit_png_depth(depth_png):
#     with Image.open(depth_png) as depth_pil:
#         # Convert to 16-bit grayscale (mode 'I' is 32-bit, so we convert properly)
#         depth_pil = depth_pil.convert("I;16")  # Ensure 16-bit depth
        
#         # Resize the shorter side to 512 while keeping aspect ratio
#         shorter_side = 512
#         width, height = depth.shape
#         aspect_ratio = width / height
#         if width < height:
#             new_width = shorter_side
#             new_height = int(shorter_side / aspect_ratio)
#         else:
#             new_height = shorter_side
#             new_width = int(shorter_side * aspect_ratio)
        
#         depth_pil = depth_pil.resize((new_width, new_height), Image.BILINEAR)

#         # Center crop to 512x512
#         left = (new_width - 512) // 2
#         top = (new_height - 512) // 2
#         depth_pil = depth_pil.crop((left, top, left + 512, top + 512))
        
#         # Convert image to numpy array
#         depth = np.array(depth_pil, dtype=np.uint16)

#         # Reinterpret as float16, then convert to float32
#         depth = depth.view(np.float16).astype(np.float32)
        
#     return depth

def load_image_as_tensor(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGB")  # Convert to RGB to ensure consistency
    
    # Define transformation to convert the image to a PyTorch tensor
    transform = transforms.Compose([
        # transforms.Resize(512, interpolation=Image.BILINEAR),  # Resize shorter side to 512
        # transforms.CenterCrop(512),  # Center crop to 512x512
        transforms.ToTensor()  # Convert to tensor with values in [0,1]    ])
    ])
    
    # Apply the transformation
    image_tensor = transform(image)
    
    return image_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="././train_configs/train_co3d_unified_nsml.yaml")
    parser.add_argument("--config", type=str, default="././train_configs/train_co3d_unified.yaml")
    # parser.add_argument("--config", type=str, default="././train_configs/train_co3d_uni_inference.yaml")
    # parser.add_argument("--config", type=str, default="././train_configs/train_co3d_unified_kaist.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)