
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from PIL import Image
from tqdm import tqdm as orig_tqdm
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage


def reprojector(pts_locations, pts_feats, camera_tgt, device, ref_depth=None, img_size=512, thresh=0.07, background=False, fov_setting="length", vggt=False, **kwargs):
    
    """
    Inverse
    Source: Unseen view
    Target: GT view
    
    """ 

    target_pose = camera_tgt["pose"]
    if not vggt:
        fovy = camera_tgt["focals"]
    else:
        fovy = camera_tgt["focals"] * (37 / 512)
        img_size = 37
        
    target_pose_inv = torch.linalg.inv(target_pose)

    batch_size = target_pose.shape[0]
        
    pts_sampled = pts_locations
        
    # if data_type == "blender" or data_type == "dtu":
        
    target_origin = target_pose[:,:3,-1]        
    target_center_viewdir = (target_pose[:,:3,2])
    
    pts_to_tgt_origin = pts_sampled - target_origin[:,None,...]
    
    dist_to_tgt_origin = torch.linalg.norm(pts_to_tgt_origin, axis=-1, keepdims=True)
    target_viewdirs = pts_to_tgt_origin / dist_to_tgt_origin

    new_per_view_lengths = (target_viewdirs * target_center_viewdir[:, None, :]).sum(axis = -1)
    target_directions = target_viewdirs / new_per_view_lengths[..., None]
    
    # Reprojector: Given target view, where do the points fall? ###############

    worldtocamera = target_pose_inv
    
    target_cameradir = (target_directions[...,None,:] * worldtocamera[..., None, :3, :3]).sum(-1)
    target_projection = target_cameradir[...,:2]

    if fov_setting == "length":
        unflip_proj = target_projection / ((img_size/2) / fovy[...,None]) # Consider focal length / FoV

    elif fov_setting == "multiple":
        unflip_proj = target_projection / fovy[:,None,:]

    projected_loc_norm = unflip_proj.reshape(-1,2).fliplr().reshape(batch_size,-1,2)
    
    proj_loc = projected_loc_norm * (img_size / 2) + (img_size / 2)
    
    # 0.5 must be added for perfect reconstruction

    proj_pix = (proj_loc + 0.5).floor()

    # print(target_pose.shape)
    # print(pts_locations.shape)
    # import pdb; pdb.set_trace()
        
    rasterized_maps, depth_maps = one_to_one_rasterizer(proj_pix, pts_feats, dist_to_tgt_origin, device, ref_depth, img_size=img_size, thresh=thresh, background=background, **kwargs)
    
    return rasterized_maps, depth_maps


def ndc_rasterizer(pts_location, pts_feats, camera_tgt, device, near=1.0, thresh=0.07, **kwargs):
    """
    NDC-space rasterizer for CO3D point cloud data.
    
    Args:
        points_3d (torch.Tensor): The point cloud (N, 3), where N is the number of points.
        features (torch.Tensor): Point cloud features (N, D), where D is the feature dimension.
        camera_intrinsics (torch.Tensor): Intrinsic matrix (3x3) of the camera.
        camera_extrinsics (torch.Tensor): Extrinsic matrix (4x4) of the camera (world to camera transformation).
        image_size (int): The size of the output image (assumed to be square).
        near (float): The near clipping plane.
        thresh (float): Threshold for distance-based filtering (optional).
    
    Returns:
        torch.Tensor: Rasterized 2D image of the point cloud features.
    """

    R = camera_tgt["rot"]
    t = camera_tgt["trans"]
    K = camera_tgt["intrinsic"]
    shape = camera_tgt["orig_img_size"]

    # import pdb; pdb.set_trace()

    batch_size = R.shape[0]

    cam_center = -torch.matmul(R, t)  # (N, 3, 1)

    # Negate the first two columns of each 3x3 matrix in R
    R[..., :2] = -R[..., :2].clone()

    # Transpose each 3x3 matrix in the batch
    R = R.transpose(-2, -1)  # Transpose the 2nd and 3rd dimensions (N, 3, 3)

    # Recalculate t using the updated R and cam_center
    t = -torch.matmul(R, cam_center)  # (N, 3, 1)

    # Concatenate R and t along the last axis
    Rt = torch.cat([R, t], dim=-1)  # (N, 3, 4)
    
    # Project
    proj_pix, pts_depth = ndc_project(pts_location, K, Rt)

    # import pdb; pdb.set_trace()

    proj_pix = proj_pix.floor().reshape(-1,2).fliplr().reshape(batch_size,-1,2)

    rasterized_maps, _ = one_to_one_rasterizer(proj_pix, pts_feats, pts_depth, device, img_size=512, **kwargs)

    return rasterized_maps


def one_to_one_rasterizer(pts_proj_pix, pts_feats, pts_depth, device, ref_depth=None, img_size=64, pts_per_pix=50, **kwargs):

    # import pdb; pdb.set_trace()    
    
    batch_size, num_pts = pts_depth.shape[0], pts_depth.shape[1]
    coord_channel = kwargs["coord_channel"]

    try:
        get_depth = kwargs["get_depth"]
    except:
        get_depth = False

    # import pdb; pdb.set_trace()

    pts_final_feats = pts_feats

    pix_key = torch.linspace(0, img_size**2 -1, steps=img_size**2).int()

    # idx_num = torch.linspace(0,num_pts-1,steps=num_pts)[None,...,None].repeat(batch_size,1,1).to(device)
    rasterizer_info = torch.cat((pts_depth, pts_final_feats),dim=-1)
    rast_bin = torch.linspace(0,pts_per_pix-1,steps=pts_per_pix).repeat(num_pts // pts_per_pix + 1)[:num_pts].int().to(device)
    
    pts_proj_pix = torch.clamp(pts_proj_pix, min=0, max=img_size-0.001).int()
    
    y_coords = pts_proj_pix[...,0]
    x_coords = pts_proj_pix[...,1]

    cnl = [0] * coord_channel
    cnl = [1] + cnl
    mult = torch.tensor(cnl)
        
    rast_map_stack = []
    depth_map_stack = []

    canv_ori = 10 * torch.ones((img_size, img_size, pts_per_pix, 1 + coord_channel)).to(device) * mult[None,None,None,...].to(device) 

    for i in range(batch_size):

        canv = canv_ori.clone()
        # import pdb; pdb.set_trace()
        canv[y_coords[i], x_coords[i], rast_bin] = rasterizer_info[i]            
                
        res_canv = canv.reshape(-1, pts_per_pix, 1 + coord_channel)
        depth_map, depth_key = torch.min(res_canv[...,0], dim=-1)
        depth_key = depth_key.flatten()
        rasterized_map = res_canv[pix_key, depth_key][:,1:].reshape(img_size, img_size, coord_channel)
        rast_map_stack.append(rasterized_map)

        if get_depth:
            depth_map = depth_map.reshape(-1,img_size,img_size)
            mask = (depth_map != 10.)

            # min_dep = torch.min(depth_map)
            depth_map = mask * depth_map
            # max_dep = torch.max(depth_map)

            # norm_dep = ((depth_map - min_dep) / (max_dep - min_dep)) * mask
            # # fin_depth = norm_dep - (1-mask.float()) # Making all empty spots -1
            # fin_depth = norm_dep + (1-mask.float()) # Making all empty spots 1

            fin_depth = depth_map
            depth_map_stack.append(fin_depth)

    rasterized_maps = torch.stack(rast_map_stack)   

    if get_depth:
        depth_maps = torch.stack(depth_map_stack)
    else:
        depth_maps = None

    return rasterized_maps, depth_maps


def ndc_project(xyz, K, RT):
    """
    Project 3D points to 2D using camera intrinsic and extrinsic parameters.
    
    Args:
        xyz (torch.Tensor): 3D points of shape [N, 3].
        K (torch.Tensor): Camera intrinsic matrix of shape [3, 3].
        RT (torch.Tensor): Camera extrinsic matrix of shape [3, 4].

    Returns:
        torch.Tensor: 2D projected points of shape [N, 2].

    """
    # Apply the camera extrinsics (RT)
    xyz_cam = torch.matmul(xyz, RT[..., :3].transpose(-2, -1)) + RT[..., 3:].transpose(-2, -1)

    # import pdb; pdb.set_trace()
    
    # Apply the camera intrinsics (K)
    xyz_proj = torch.matmul(xyz_cam, K.transpose(-2, -1))
    
    # Normalize by z to get 2D coordinates
    depth =  xyz_proj[..., 2:].clamp(min=1e-5)
    xy = xyz_proj[..., :2] / depth  # Avoid division by zero

    return xy, depth
    