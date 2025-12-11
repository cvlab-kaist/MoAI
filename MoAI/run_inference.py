import argparse
import logging
import math
import json
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

import transformers
import cv2
import multiprocessing as mp
import inspect

import lpips  # pip install lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

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

from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image

from main.models.resnet import InflatedConv3d, InflatedGroupNorm
from main.models.mutual_self_attention import ReferenceAttentionControl
from main.models.pose_guider import PoseGuider
from main.models.unet_2d_condition import UNet2DConditionModel
from main.models.unet_3d import UNet3DConditionModel
from main.models.hook import UNetCrossAttentionHooker, XformersCrossAttentionHooker
from main.pipelines.pipeline_nvs import NVSPipeline
from training_utils import delete_additional_ckpt, import_filename, seed_everything, load_model
from einops import rearrange

from training_utils import forward_warper, camera_controller, plucker_embedding, get_embedder, get_coords
from torchvision.transforms.functional import to_pil_image

from main.utils.attn_visualizer import sample_zero_mask_pixels, get_attn_map, stitch_side_by_side

from main.utils import (
    reprojector,
    mesh_rendering,
    features_to_world_space_mesh,
    torch_to_o3d_cuda_mesh,
    PointmapNormalizer,
    EvalBatch,
    camera_search,
    overlay_grid_and_save,
    mari_embedding_prep
)

from main.ops import (
    make_video,
    find_closest_camera,
    depth_normalize,
    depth_metrics,
    closed_form_inverse_se3,
    encode_depth,
    mesh_get_depth,
    apply_heatmap,
    convert_depth_to_normal,
    convert_opencv_extrinsics_to_view,
    camera_lookat
)


import torchvision.transforms as transforms
from einops import rearrange, repeat
from camera_visualization import pointmap_vis
from transformers import CLIPImageProcessor


warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

os.environ["OPEN3D_RENDERING_BACKEND"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime"
if not os.path.exists("/tmp/runtime"):
    os.makedirs("/tmp/runtime", mode=0o700)
    

class MoAI(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        geometry_unet: UNet3DConditionModel,
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
        timesteps,
        ref_image_latents,
        geo_ref_latents,
        clip_image_embeds,
        ref_coord_embeds,
        tgt_coord_embed,
        correspondence = None,
        weight_dtype = None,
        task_emb = None,
        attn_proc_hooker=None,
        geo_attn_proc_hooker=None,
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
                # ref_latent,
                geo_ref_latents[i],
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
        
        clip_closest_embeds = []
        for batch_num in range(clip_image_embeds.shape[1]):
            clip_closest_embeds.append(clip_image_embeds[closest_idx[batch_num], batch_num])
        tgt_clip_embed = torch.stack(clip_closest_embeds)
                    
        extra_step_kwargs = prepare_extra_step_kwargs(val_scheduler)
        
        input_dict = {
            "img_pred": noisy_latents,
            "geo_pred": geo_noisy_latents
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
            
            # Add warped_image_latent
            if n != 0:
                geo_noisy_latents = torch.cat([warped_image_latents, geo_noisy_latents], dim=1)
            geo_latent_model_input = torch.cat([geo_noisy_latents] * 2)
            geo_latent_model_input = val_scheduler.scale_model_input(
                    geo_latent_model_input, t
                )
            
            key = int(t)
            attn_proc_hooker.cross_attn_maps[key] = []
            attn_proc_hooker.current_timestep = key
            
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

            geo_attn_proc_hooker.layer_list = attn_proc_hooker.layer_list.copy()
                        
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
    normalize_coord=True,
    tgt_image=None,
    ref_depth=None,
    confidence=None,
    warp_image=True,
    mesh_pts=None,
    mesh_depth=None,
    mesh_normals=None,
    mesh_ref_normals=None,
    normal_mask=None,
    plucker=None,
    pts_norm_func=None,
    current_dataset="multi",
    full_condition=False,
):
    
    # Prepare inputs.
    src_image = src_images[0]
    B = src_image.shape[0] # Image shape (B, num_views, 3, H, W)
    num_ref_views = correspondence["ref"].shape[1]
    H, W = src_image.shape[-2:]
    device = mesh_normals.device

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
                pts_feat = torch.cat((mod_pts_locs, pts_rgb.to(device)), dim=-1)
            else:
                pts_feat = mod_pts_locs

            coord_channel = pts_feat.shape[-1]
            camera_tgt = camera_info["tgt"]

            combined_results, tgt_depth = reprojector(pts_locs, pts_feat, camera_tgt, device=device, coord_channel=coord_channel, get_depth=depth_conditioning)
            proj_results = combined_results[...,:3]

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
        
    else:
        mesh_pts = mesh_pts.to(device)
        mesh_depth = mesh_depth.to(device)

        mask = (mesh_pts != 0)
        pts_locs = correspondence["ref"].reshape(B,-1,3).to(device)
        
        if pts_norm_func is not None:
            src_corr = pts_norm_func(correspondence["ref"].to(device))
            proj_results = mask * pts_norm_func(mesh_pts)
            
        else:
            max_val = torch.max(pts_locs, dim=-2)[0].to(device)
            min_val = torch.min(pts_locs, dim=-2)[0].to(device)
            
            src_corr = ((correspondence["ref"].to(device) - min_val[:,None,None,None,:]) / (max_val[:,None,None,None,:] - min_val[:,None,None,None,:])) * 2 - 1
            proj_results = mask * (((mesh_pts - min_val[:,None,None,:]) / (max_val[:,None,None,:] - min_val[:,None,None,:])) * 2 - 1 )
            
        tgt_depth = mesh_depth.unsqueeze(1)

        if warp_image:
            pts_rgb = torch.cat(src_images,dim=1).permute(0,1,3,4,2).reshape(B,-1,3)
            camera_tgt = camera_info["tgt"]
            image_warped, _ = reprojector(pts_locs, pts_rgb, camera_tgt, device=device, coord_channel=3, get_depth=False)
            warped_results = image_warped
    
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

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_params)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        kwargs_handlers=[kwargs],
    )    

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
    
    val_scheduler = DDIMScheduler(**sched_kwargs)
    val_scheduler.set_timesteps(
        20, device="cuda")

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    cond_channels = 16
    if cfg.use_depthmap:
        cond_channels += 1
    if cfg.use_normal:
        cond_channels += 3
    
    device, dtype = "cuda", weight_dtype
    num_viewpoints = cfg.dataset.num_viewpoints
    num_ref_viewpoints = cfg.dataset.num_ref
        
    def init_2d(model_cls, config_fname, weight_fname, **init_kwargs):
        cfg_path = join(cfg.model_path, config_fname)
        model = model_cls.from_config(model_cls.load_config(cfg_path), **init_kwargs)
        model.to(device=device, dtype=dtype)
        model.load_state_dict(torch.load(
            join(cfg.model_path, weight_fname),
            map_location="cpu"
        ))
        return model

    def init_3d(model_cls, config_fname, weight_fname):
        return model_cls.from_pretrained_2d(
            join(cfg.model_path, config_fname),
            join(cfg.model_path, weight_fname)
        ).to(device=device, dtype=dtype)

    pose_guider           = PoseGuider(320, cond_channels).to(device=device, dtype=dtype)
    reference_unet        = init_2d(UNet2DConditionModel, "configs/image_config.json",         "reference_unet.pth")
    denoising_unet        = init_3d(UNet3DConditionModel, "configs/image_config.json",         "denoising_unet.pth")
    geometry_unet         = init_3d(UNet3DConditionModel, "configs/geometry_config.json", "geometry_unet.pth")
    geo_reference_unet    = init_2d(UNet2DConditionModel, "configs/geometry_config.json", "geo_reference_unet.pth")

    pose_guider.load_state_dict(torch.load(
        join(cfg.model_path, 'pose_guider.pth'),
        map_location='cpu'),
    )
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    clip_preprocessor = CLIPImageProcessor()
    embedder, out_dim = get_embedder(2)

    # Freeze
    for m in (
        vae,
        image_enc,
        denoising_unet,
        reference_unet,
        pose_guider,
        geometry_unet,
        geo_reference_unet,
    ):
        m.requires_grad_(False)
            
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
    
    net_variables = [
        reference_unet,
        denoising_unet,
        geometry_unet,
        geo_reference_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
        geo_reference_control_writer,
        geo_reference_control_reader
    ]    
    
    net = MoAI(
        *net_variables
    )
    
    if cfg.inference:
        net.inference = True
    
    logger.info(f"Feature fusion type is '{cfg.feature_fusion_type}'")

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
            geometry_unet.enable_xformers_memory_efficient_attention()
            geo_reference_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
            
    attn_proc_hooker=XformersCrossAttentionHooker(True, num_ref_views=num_ref_viewpoints, setting="writing", cross_attn=cfg.use_geo_ref_unet, save_attn_map=False)    
    denoising_unet.set_attn_processor(attn_proc_hooker)
    geo_attn_proc_hooker=XformersCrossAttentionHooker( True, num_ref_views=num_ref_viewpoints, setting="reading", cross_attn=cfg.use_geo_ref_unet)
    geometry_unet.set_attn_processor(geo_attn_proc_hooker)
         
    # Pointmap embedder initialization
    if cfg.embed_pointmap_norm:        
        if cfg.train_vggt:
            ptsmap_min = torch.tensor([-0.6338, -0.4921, 0.4827]).to(image_enc.device)
            ptsmap_max = torch.tensor([ 0.6190, 0.6307, 1.6461]).to(image_enc.device)            
        else:   
            ptsmap_min = torch.tensor([-0.1798, -0.2254,  0.0593]).to(image_enc.device)
            ptsmap_max = torch.tensor([0.1899, 0.0836, 0.7480]).to(image_enc.device)
            
        pts_norm_func = PointmapNormalizer(ptsmap_min, ptsmap_max, k=0.9)
        
    device = image_enc.device            
    now = strftime("%m_%d_%H_%M_%S", gmtime())
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    eval_batchify = EvalBatch(device, recon_model = "DepthAnythingV3")
    batch, num_ref_viewpoints = eval_batchify.images_eval(image_dir=cfg.eval_image_dir)
    src_idx = [i for i in range(num_ref_viewpoints)]
    target_idx_list = [num_ref_viewpoints-1]
        
    instance_now = strftime("%m_%d_%H_%M_%S", gmtime())
    pointmap_list = []
    target_pose_list = []

    for target_idx in target_idx_list:
        
        images = dict(ref=[batch["image"][:,k].unsqueeze(1) for k in src_idx], tgt=batch["image"][:,target_idx])

        with torch.no_grad():
            batch_size, num_view, _, _ = batch['extrinsic'].shape
            pointmaps = batch['points'].permute(0,1,3,4,2)
            w2c = torch.cat((batch['extrinsic'], torch.tensor([0,0,0,1]).to(device)[None,None,None,...].repeat(batch_size, num_view, 1, 1)), dim=-2)
            orig_extrinsic = torch.linalg.inv(w2c)
                        
            if cfg.normalized_pose:
                extrinsic = torch.matmul(w2c[:,target_idx].unsqueeze(1), orig_extrinsic)
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
            current_dataset = "realestate"

            # View selection from reference camera
            search = True
                                        
            pts_locs = correspondence["ref"].reshape(1,-1,3)
            pts_rgb = torch.cat(images['ref'],dim=1)[0].permute(0,2,3,1).reshape(1,-1,3)
            search_cam = tgt_camera.copy()
            
            # Initial rendering                  
            rendering, _ = reprojector(pts_locs, pts_rgb, search_cam, device=device, coord_channel=3, get_depth=False)
            
            overlay_grid_and_save(rendering.permute(0,3,1,2)[0], out_path="RENDERING.png")
                                        
            while search:
                movement_cmd = input("Cmd [W/A/S/D translate, T/F/G/H rotate, END to finish]: ").strip().upper()
                
                cam_pose = search_cam["pose"]
                search_cam["pose"] = camera_search(cam_pose[0], movement_cmd, device)
                
                rendering, _ = reprojector(pts_locs, pts_rgb, search_cam, device=device, coord_channel=3, get_depth=False)
                overlay_grid_and_save(rendering.permute(0,3,1,2)[0], out_path="RENDERING.png")
                
                ask_continue = input("Continue searching, or no? : ")
                
                if ask_continue == "no":
                    search = False
                else:
                    pass
                
            tgt_camera["pose"] = search_cam["pose"]  
            camera_info["tgt"] = tgt_camera
            
            target_pose_list.append(torch.matmul(orig_extrinsic[:,target_idx],search_cam["pose"]))

            tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth = mari_embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, batch_size, device, full_condition=full_condition)
            src_images = images["ref"]
            
            args = dict(
                src_images=src_images,
                correspondence=correspondence,
                camera_info=camera_info,
                embedder=embedder,
                src_idx=src_idx,
                tgt_idx=target_idx,
                tgt_image=images["tgt"],
                current_dataset=current_dataset
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
            
            # Image latent preparation
            conditions, renders = prepare_duster_embedding(**args)
            latents = vae.encode(images["tgt"].to(weight_dtype) * 2 - 1).latent_dist.sample()
            
            # Geometry latent preparation
            ref_pointmaps = correspondence["ref"].reshape(-1,512,512,3).permute(0,3,1,2)
            tgt_pointmap = correspondence["tgt"].permute(0,3,1,2)
            
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
                    
            geo_latents = vae.encode(tgt_pointmap.to(weight_dtype)).latent_dist.sample()
            
                                    
            geo_latents = geo_latents.unsqueeze(2) 
            geo_latents = geo_latents * 0.18215  
            
            latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
            latents = latents * 0.18215    

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
                    ref_pointmaps,
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
                        
        initial_t = torch.tensor(
            [999] * batch_size
        ).to(device, dtype=torch.long)
            
        latents_noisy_start = val_scheduler.add_noise(
            latents, noise, initial_t
        )
        
        noisy_latents = latents_noisy_start
        geo_noisy_latents = latents_noisy_start

        # Image embeddings ------------------
        image_prompt_embeds = image_prompt_embeds.reshape(B,V,1,-1).permute(1,0,2,3)
        ref_image_latents = ref_image_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
        task_emb = None
        
        if cfg.use_warped_img_cond:
            noisy_latents = torch.cat((noisy_latents, warped_latents), dim=1)
        
        # Geometry embeddings ------------ (always have warped latents as condition)
        geo_noisy_latents = torch.cat((warped_latents, geo_noisy_latents), dim=1)
        
        # Reference network for geometry
        if cfg.use_geo_ref_unet:
            ref_depth_latents = ref_depth_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
            geo_ref_latents = torch.cat((ref_image_latents, ref_depth_latents), dim=2).to(weight_dtype)
        else:
            geo_ref_latents = None
                            
        with torch.no_grad():
            results_dict = net(
                noisy_latents.to(weight_dtype),
                geo_noisy_latents.to(weight_dtype),
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
                closest_idx=closest_idx,
                val_scheduler=val_scheduler,
                vae = vae
            )
            
            # Directory for this inference                      
            exp_name = cfg.inference_run_name
            dir = f"{exp_name}/{now}/{instance_now}"
            
            if not os.path.exists(dir):
                os.makedirs(dir)
                                        
            # Metrics
            depth_map = results_dict['geo_pred'][:,2:].repeat(1,3,1,1)       
            normal = convert_depth_to_normal(depth_map[:,:1])  
            gt_normal = convert_depth_to_normal(tgt_pointmap[:,2:] * 0.5 + 0.5)

            depth = apply_heatmap((depth_map[:,:1] != 0)* 1 / depth_map[:,:1]).to(device)   
            gt_depth = apply_heatmap( ((tgt_pointmap[:,2:] * 0.5 + 0.5)!=0) * 1 / (tgt_pointmap[:,2:] * 0.5 + 0.5)).to(device)  

            ref_images = torch.cat(images["ref"]).squeeze()
                                                                
            # Saving the images
            space = torch.ones(1,3,512,80).to(device)
            large_space = torch.ones(1,3,512,420).to(device)
                                        
            for i, ref in enumerate(ref_images):
                save_image(ref, f"{dir}/ref_{i}.png")  
                                            
            stack_img = torch.cat((images["tgt"].to(device), space, zero_mask * (warped_image * 0.5 + 0.5), space, tgt_pointmap*0.5+0.5, space, gt_depth, space, results_dict['img_pred'], space, results_dict['geo_pred'], space, depth, space, normal * 0.5 + 0.5), dim=-1).detach()[0]                    
            save_image(stack_img, f"{dir}/target_stack_{target_idx}.png")
            
            warp = zero_mask * mesh_normal_mask.permute(0,3,1,2) * (warped_image * 0.5 + 0.5)
            warp_depth = mesh_normal_mask.permute(0,3,1,2) * apply_heatmap((renders['tgt_depth'] != 0) * (1 / renders['tgt_depth'])).to(device)
            
            save_image(warp, f"{dir}/warped_tgt_{target_idx}.png")
            save_image(warp_depth, f"{dir}/warped_depth_tgt_{target_idx}.png")
            
            if cfg.geo_first:
                val_img = torch.cat((zero_mask * (warped_image * 0.5 + 0.5), space, results_dict['img_pred'], space, depth, space, normal * 0.5 + 0.5, space, space, large_space, large_space, space, images["tgt"].to(device), space, gt_depth, space, gt_normal * 0.5 + 0.5), dim=-1).detach()[0]                                      
                save_image(val_img, f"{dir}/target_view_{target_idx}.png")
                                            
                if minmax_set:
                    pts_loc = (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * (results_dict['geo_pred']) + ptsmap_min[None,...,None,None]
                else:
                    pts_loc = (ptsmap_max[...,None,None] - ptsmap_min[...,None,None]) * (results_dict['geo_pred']) + ptsmap_min[...,None,None]
                pts_rgb = results_dict['img_pred']

                points = torch.cat((pts_loc, pts_rgb), dim=1).permute(0,2,3,1).reshape(B,-1,6)
                pts_map = torch.matmul(orig_extrinsic[:,target_idx][:,None,:3,:3], (points[...,:3] - w2c[:,target_idx][:,None,:3,3]).unsqueeze(-1))             
                                            
                pointmap_list.append(torch.cat((pts_map.squeeze(-1), points[...,3:]),dim=-1))     
                
                if target_idx == target_idx_list[-1]:
                    
                    if cfg.save_everything or cfg.infer_setting=="view_search" or cfg.infer_setting=="revisit":
                        
                        torch.save(batch, f"{dir}/batch_info.pt")
                    
                        ref_pts_rgb = ref_images                                
                        ref_points = torch.cat((batch['points'][0,src_idx], ref_pts_rgb),dim=1).permute(0,2,3,1).reshape(1,-1,6)
                        
                        torch.save(ref_points, f"{dir}/ref_pts.pt")
                        
                        pointmap_stack = torch.stack(pointmap_list, dim=0).to(device)  
                        torch.save(pointmap_stack, f"{dir}/all_pts.pt")
                        
                    if cfg.view_select:
                        tgt_pose_stack = torch.cat(target_pose_list)
                    
                    camera_dict = {
                        "ref_ext": orig_extrinsic[:,src_idx][0],
                        "ref_focal":ref_camera['focals'][0],
                        "tgt_ext": orig_extrinsic[:,target_idx_list][0] if not cfg.view_select else tgt_pose_stack,
                        "tgt_focal": focal[:,target_idx_list][0]
                    }
                    
                    torch.save(camera_dict, f"{dir}/camera_info.pt")
                    
        attn_proc_hooker.clear()
        attn_proc_hooker.attn_map_clear()                                               
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="././eval_configs/eval.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)