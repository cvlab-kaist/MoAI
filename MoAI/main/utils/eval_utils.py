import os
import glob
import torch
import numpy as np
import argparse
import math
import json

import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from pathlib import Path

import torch.nn.functional as F
import torchvision.transforms as tf

from torchvision.utils import save_image
from dataclasses import asdict, dataclass

from vggt_original.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_original.models.vggt import VGGT
from torchvision.utils import save_image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

from dacite import Config, from_dict
from jaxtyping import Float, Int64

@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]
    
def pil_to_tensor_batch(pil_images):
    """
    Convert a list of PIL.Image to a 4D tensor batch of shape (B, C, H, W).
    
    Args:
        pil_images (List[PIL.Image.Image]): List of PIL images (mode "RGB").

    Returns:
        torch.Tensor: A tensor of shape (B, 3, H, W) with pixel values in [0,1].
    """
    to_tensor = ToTensor()  # Converts PIL Image to (C, H, W) tensor in [0,1]
    tensors = [to_tensor(img) for img in pil_images]
    batch = torch.stack(tensors, dim=0)  # Stack into (B, C, H, W)
    return batch

class EvalBatch():
    def __init__(
            self,
            device,
            recon_model = "DepthAnythingV3"   
        ):
        self.chunk_num = 0
        self.example_num = 0
        self.device = device
        self.dtype = torch.bfloat16  # or torch.float16

        if recon_model == "DepthAnythingV3": 
            self.model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device)
            self.model.eval()
            self.model_name = "da3"
            
        elif recon_model == "vggt":
            self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
            self.mode_name = "vggt"
        
        self.crop_transform = tf.Compose([
            tf.Resize(512, interpolation=tf.InterpolationMode.BICUBIC),  # Resize shortest side to 512
            tf.CenterCrop(512),  # Center crop to 512x512
            tf.ToTensor()    
        ])

    
    def transform_pts(self, array):
        """
        Transform a NumPy array image by:
        1. Resizing so the shortest side is 512 pixels (bicubic interpolation).
        2. Center cropping to 512x512.
        3. Normalizing pixel values to [0, 1].
        
        Args:
            image (np.ndarray): Input image in shape (H, W, C) or (H, W).
        
        Returns:
            np.ndarray: Transformed image.
        """
        # Convert numpy array to PIL Image
        # Determine new size keeping the aspect ratio,
        # so that the shortest side becomes 512.
        array = array.permute(0,3,1,2)
        _, _, height, width = array.shape
        
        if width < height:
            new_width = 512
            new_height = int(512 * height / width)
        else:
            new_height = 512
            new_width = int(512 * width / height)

        # Resize image using bicubic interpolation
        array_resized = F.interpolate(array, size=(new_height, new_width), mode="bilinear")
        # im_resized = im.resize((new_width, new_height), resample=Image.BICUBIC)

        # Compute coordinates for center crop of 512x512
        left = (new_width - 512) // 2
        top = (new_height - 512) // 2
        right = left + 512
        bottom = top + 512
        

        # Center crop the image
        array = array_resized[...,top:bottom,left:right]
        return array

    def images_eval(self, image_dir = "/path/to/your/image_directory"):
                
        # find all common image files, sorted by name
        image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*"))
        )
        # filter out non-image extensions
        image_paths = [
            p for p in image_paths
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        # open each file and convert to RGB PIL.Image
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        
        num_ref_views = len(pil_images)
        images = pil_to_tensor_batch(pil_images)
        
        with torch.no_grad():
            if self.model_name == "vggt"
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # Time aggregator call (including adding the batch dimension)
                    images = images[None]  # add batch dimension
                    images_vggt = images_vggt[None]  # add batch dimension
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images_vggt.to(self.device))
                
                # Time the camera head prediction
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                
                # Time the conversion from pose encoding to extrinsic/intrinsic matrices
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_vggt.shape[-2:])
                
                # Time the point head prediction
                point_map, point_conf = self.model.point_head(aggregated_tokens_list, images_vggt, ps_idx)
            
                pts = self.transform_pts(point_map[0])
                images = self.transform_pts(images[0].permute(0,2,3,1))
            
            elif self.model_name == "da3"
                images = self.transform_pts(images.permute(0,2,3,1), size=504).permute(0,3,1,2)  
                da3_imgs = [image.permute(1,2,0).detach().to("cpu").numpy() for image in images] 
                
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    prediction = self.model.inference(
                            image=da3_imgs,
                            process_res=504
                        )    

                extrinsic = prediction.extrinsics
                intrinsic = prediction.intrinsics
                depth_map = prediction.depth

                extrinsic = torch.tensor(extrinsic).unsqueeze(0).to(device)
                intrinsic = torch.tensor(intrinsic).unsqueeze(0).to(device)

                point_map = unproject_depth_map_to_point_map(depth_map[...,None], extrinsic[0], intrinsic[0])

                # import pdb; pdb.set_trace()
                pts=self.transform_pts(torch.tensor(point_map), size=512).permute(0,3,1,2)
                images = self.transform_pts(images.permute(0,2,3,1), size=512).permute(0,3,1,2)   
            
            output = dict(image = images[None,...].to(self.device), points = pts[None,...].to(self.device), intrinsic = intrinsic.to(self.device), extrinsic = extrinsic.to(self.device), conf=None, dataset=[1])
        
        return output, num_ref_views


def overlay_grid_and_save(img_tensor: torch.Tensor, spacing=64, out_path="grid_overlay.png"):
    """
    Overlays a grid every `spacing` pixels, but only labels the
    x-axis at the top edge and the y-axis at the left edge,
    with labels in black. The image is shown without flipping.
    
    img_tensor: [H, W] or [C, H, W]
    """
    # 1. Convert to H×W or H×W×C numpy
    if img_tensor.ndim == 3:
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
    elif img_tensor.ndim == 2:
        img = img_tensor.cpu().numpy()
    else:
        raise ValueError(f"Unsupported shape {img_tensor.shape}")

    H, W = img.shape[:2]
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    # 2. Plot
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=None if img.ndim == 3 else "gray", origin="upper")
    # — no ax.invert_xaxis(), so image is not flipped

    # 3. Draw grid lines
    for y in ys:
        ax.axhline(y, color="white", linewidth=0.8)
    for x in xs:
        ax.axvline(x, color="white", linewidth=0.8)

    # 4. Set ticks at grid lines
    ax.set_xticks(xs)
    ax.set_yticks(ys)

    # 5. Label ticks in black
    ax.set_xticklabels([str(x) for x in xs], color="black", fontsize=8)
    ax.set_yticklabels([str(y) for y in ys], color="black", fontsize=8)

    # 6. Show x labels on top only, y labels on left only
    ax.tick_params(
        axis='x', which='both',
        labelbottom=False,
        labeltop=True,
        bottom=False, top=False
    )
    ax.tick_params(
        axis='y', which='both',
        labelleft=True,
        labelright=False,
        left=False, right=False
    )

    # 7. Remove frame lines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)     

def make_translation_matrix(dx, dy, dz, device):
    T = torch.eye(4)
    T[0,3] = dx
    T[1,3] = dy
    T[2,3] = dz
    return T.to(device=device)

def make_rotation_matrix(axis, angle_deg, device):
    θ = math.radians(angle_deg)
    c, s = math.cos(θ), math.sin(θ)
    R = torch.eye(4)
    if axis == 'x':
        R[1,1], R[1,2] = c, -s
        R[2,1], R[2,2] = s,  c
    elif axis == 'y':
        R[0,0], R[0,2] =  c, s
        R[2,0], R[2,2] = -s, c
    elif axis == 'z':
        R[0,0], R[0,1] = c, -s
        R[1,0], R[1,1] = s,  c
    return R.to(device=device)

def camera_search(cam, cmd, device):
    # start from identity
    t_step = 0.15
    r_step = 10.0

    if cmd == 'W':
        T = make_translation_matrix(0, 0, t_step, device)
        cam = T @ cam
    elif cmd == 'S':
        T = make_translation_matrix(0, 0, -t_step, device)
        cam = T @ cam
    elif cmd == 'A':
        T = make_translation_matrix(-t_step, 0, 0, device)
        cam = T @ cam
    elif cmd == 'D':
        T = make_translation_matrix( t_step, 0, 0, device)
        cam = T @ cam
    elif cmd == 'T':  # pitch up  (rotate around x)
        R = make_rotation_matrix('x',  r_step, device)
        cam = R @ cam
    elif cmd == 'G':  # pitch down
        R = make_rotation_matrix('x', -r_step, device)
        cam = R @ cam
    elif cmd == 'F':  # yaw left (rotate around y)
        R = make_rotation_matrix('y',  -r_step, device)
        cam = R @ cam
    elif cmd == 'H':  # yaw right
        R = make_rotation_matrix('y', r_step, device)
        cam = R @ cam
    else:
        print("Unknown command.")

    return cam[None,...]

 
@torch.no_grad()
def mari_embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, batch_size, device, full_condition=False):
    mesh_pts = None
    mesh_depth = None
    mesh_normals = None
    mesh_ref_normals = None
    mesh_normal_mask = None
    norm_depth = None
    confidence_map = None
    plucker = None 
    tgt_depth = None
    
    origins = ref_camera['pose'][:,:,:3,-1]
    tgt_origins = tgt_camera['pose'][:,:3,-1]
    
    dist = torch.linalg.norm(correspondence["ref"] - origins[...,None,None,:],axis=-1)[...,None]
    norm_depth = dist
    
    tgt_dist = torch.linalg.norm(correspondence["tgt"] - tgt_origins[...,None,None,:],axis=-1)[...,None]
    tgt_depth = tgt_dist
                                
    ref_depth = depth_normalize(cfg, norm_depth)
    tgt_depth_norm = depth_normalize(cfg, tgt_depth)
    
    clip = True
                                
    if clip:
        min_val = -1.0
        max_val = 1.0
        ref_depth = torch.clip(ref_depth, min=min_val, max=max_val).reshape(-1,512,512,1).permute(0,3,1,2).repeat(1,3,1,1)
        tgt_depth_norm = torch.clip(tgt_depth_norm, min=min_val, max=max_val).permute(0,3,1,2).repeat(1,3,1,1)
    
    if cfg.downsample:
        downsample_by = cfg.downsample_by
        start = downsample_by // 2
        interval = downsample_by
        if not full_condition:
            points = correspondence['ref'][:,:,start::interval,start::interval,:].permute(0,1,4,2,3).float()    
        else:
            ref_points = correspondence['ref'][:,:,start::interval,start::interval,:].permute(0,1,4,2,3).float() 
            tgt_points = correspondence['tgt'][:,start::interval,start::interval,:].unsqueeze(1).permute(0,1,4,2,3).float() 
            
            points = torch.cat((ref_points, tgt_points), dim=1)

        images_ref = torch.cat(images["ref"], dim=1)
        rgb = images_ref[:,:,:,start::interval,start::interval]
        if full_condition:
            tgt_rgb = images["tgt"][:,:,start::interval,start::interval].unsqueeze(1)
            rgb = torch.cat((rgb, tgt_rgb), dim=1)
        side_length = 512 // downsample_by

    batch_pts = points
    batch_colors = rgb
    orig_length = torch.tensor(512).to(device)

    mesh_pts = []
    mesh_normals = []
    mesh_depth = []
    mesh_ref_normal_list = []
    mesh_normal_mask = []

    for i, (pts_list, color_list) in enumerate(zip(batch_pts, batch_colors)):
        extrins = tgt_camera["pose"][i].detach().cpu().numpy()
        focal_length = tgt_camera["focals"][i]

        vert = []
        fc = []
        col = []

        vert_stack = 0

        for k, (pts, color) in enumerate(zip(pts_list, color_list)):
            
            if not full_condition:
                vertices, faces, colors = features_to_world_space_mesh(
                    world_space_points=pts.detach(),
                    colors=color.detach(),
                    edge_threshold=0.48,
                    H = side_length
                )

                vert.append(vertices)
                fc.append(faces + vert_stack)
                col.append(colors)

                vert_num = vertices.shape[1]
                vert_stack += vert_num

            else:
                if k != (pts_list.shape[0] - 1):
                    vertices, faces, colors = features_to_world_space_mesh(
                        world_space_points=pts.detach(),
                        colors=color.detach(),
                        edge_threshold=0.48,
                        H = side_length
                    )

                    vert.append(vertices)
                    fc.append(faces + vert_stack)
                    col.append(colors)

                    vert_num = vertices.shape[1]
                    vert_stack += vert_num
                else:
                    tgt_vertices, tgt_faces, tgt_colors = features_to_world_space_mesh(
                        world_space_points=pts.detach(),
                        colors=color.detach(),
                        edge_threshold=0.48,
                        H = side_length
                    )
                                        
        vertices = torch.cat(vert, dim=-1)
        faces = torch.cat(fc, dim=-1)
        colors = torch.cat(col, dim=-1)

        if full_condition:
            tgt_mesh, o3d_device = torch_to_o3d_cuda_mesh(tgt_vertices, tgt_faces, tgt_colors, device = pts.device)
            mesh, _ = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)
            
        else:
            tgt_mesh, o3d_device = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)
            mesh = tgt_mesh

        inv_extrins = np.linalg.inv(extrins)
        rendered_depth, normals = mesh_rendering(tgt_mesh, focal_length, inv_extrins, o3d_device)
        rays_o, rays_d = get_rays(orig_length, orig_length, focal_length.to(device), torch.tensor(extrins).to(device), 1, device)
        mask = (rendered_depth != 0)

        proj_pts = mask[...,None].to(device) * (rays_o[0,0] + rendered_depth[...,None].to(device) * rays_d[0,0])

        if cfg.use_normal_mask:
            center_dir = rays_d[0, 0, 256, 256][None,None,...] 

            normed_center_dir = -center_dir / torch.norm(center_dir, dim=-1, keepdim=True) 
            normed_normals = normals.to(device) / torch.norm(normals, dim=-1, keepdim=True).to(device)

            dot_product = torch.clamp(torch.sum(normed_center_dir * normed_normals, dim=-1, keepdim=True), -1.0, 1.0) 
            angle_difference = torch.acos(dot_product)

            angle_mask = angle_difference > (torch.pi * 1/2)
            mesh_normal_mask.append(angle_mask)

        mesh_pts.append(proj_pts)
        mesh_depth.append(rendered_depth)
        mesh_normals.append(normals)

        if cfg.use_normal:
            per_batch_ref_normals = []

            for ref_extrins, ref_focal in zip(ref_camera["pose"][i], ref_camera["focals"][i]):
                ref_extrins = ref_extrins.detach().cpu().numpy()
                ref_pose = np.linalg.inv(ref_extrins)
                ref_depths, ref_normals = mesh_rendering(mesh, ref_focal, ref_pose, o3d_device)
                per_batch_ref_normals.append(ref_normals.permute(2,0,1))
            
            mesh_ref_normals = torch.stack(per_batch_ref_normals)
            mesh_ref_normal_list.append(mesh_ref_normals)
        
    mesh_pts = torch.stack(mesh_pts)
    mesh_depth = torch.stack(mesh_depth).to(device)
    mesh_normals = torch.stack(mesh_normals)

    if cfg.use_normal:
        mesh_ref_normals = torch.stack(mesh_ref_normal_list)

    if cfg.use_normal_mask:
        # save_image(torch.cat((mesh_normals.permute(0,3,1,2).to(device), mesh_normal_mask[0][None,...].permute(0,3,1,2).repeat(1,3,1,1))), "new.png")

        mesh_normal_mask = (1 - torch.stack(mesh_normal_mask).float()).to(device)

        mesh_pts = mesh_pts * mesh_normal_mask
        mesh_depth = mesh_depth * mesh_normal_mask[...,0]
        mesh_normals = mesh_normals.to(device) * mesh_normal_mask
            
    return tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth
