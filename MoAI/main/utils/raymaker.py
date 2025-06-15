import torch

def get_rays(H, W, focals, c2w, batch_size, device):
    """
    Get ray origins and directions from a pinhole camera model in PyTorch.

    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length of the camera.
        c2w (torch.Tensor): Camera-to-world transformation matrix of shape (4, 4) or (3, 4).
        
    Returns:
        rays_o (torch.Tensor): Ray origins of shape (H, W, 3).
        rays_d (torch.Tensor): Ray directions of shape (H, W, 3).
    """
    # Create meshgrid for image coordinates (i, j)

    # short_idx = torch.min(torch.tensor([H,W]))
    # factor = 512 * max(H,W)/min(H,W)

    ray_len = 512
    short_side = torch.min(H,W)

    ray_W = ray_len * W / short_side
    ray_H = ray_len * H / short_side

    margin_W = ray_W / 2 - ray_len / 2
    margin_H = ray_H / 2 - ray_len / 2

    i, j = torch.meshgrid(torch.arange(ray_len, dtype=torch.float32), torch.arange(ray_len, dtype=torch.float32), indexing='xy')

    i = i.to(device)
    j = j.to(device)

    # Compute directions (normalized by focal length)
    focals = focals.reshape(-1,1)

    view_num = focals.shape[0]

    dirs_stack = []

    i = i[None,None,...]
    j = j[None,None,...]

    dirs = torch.stack([(i - ray_W * 0.5 + margin_W) / focals[None,...,None], (j - ray_H * 0.5 + margin_H) / focals[None,...,None], torch.ones_like(i.repeat(batch_size,view_num,1,1))], dim=-1)  # Shape (H, W, 3)

    # Apply camera-to-world rotation matrix to directions
    rays_d = torch.sum(dirs[..., None, :] * c2w[..., None, None, :3, :3], dim=-1)  # Shape (H, W, 3)

    # Broadcast ray origins to match shape (H, W, 3)
    rays_o = c2w[..., None, None, :3, -1].expand(rays_d.shape)  # Shape (H, W, 3)

    return rays_o, rays_d



# def get_rays(H, W, focals, c2w, batch_size, device):
#     """
#     Get ray origins and directions from a pinhole camera model in PyTorch.

#     Args:
#         H (int): Image height.
#         W (int): Image width.
#         focals (torch.Tensor): Focal lengths of the camera (B,).
#         c2w (torch.Tensor): Camera-to-world transformation matrix of shape (B, 4, 4).
#         batch_size (int): Number of cameras in the batch.
#         device (torch.device): Device to run the computations on.
        
#     Returns:
#         rays_o (torch.Tensor): Ray origins of shape (B, ray_len*ray_len, 3).
#         rays_d (torch.Tensor): Ray directions of shape (B, ray_len*ray_len, 3).
#     """
#     ray_len = 512  # Fixed resolution for the rays
#     short_side = min(H, W)
#     ray_W = ray_len * W / short_side
#     ray_H = ray_len * H / short_side
#     margin_W = ray_W / 2 - ray_len / 2
#     margin_H = ray_H / 2 - ray_len / 2

#     # Generate normalized image plane grid
#     i, j = torch.meshgrid(
#         torch.arange(ray_len, dtype=torch.float32, device=device),
#         torch.arange(ray_len, dtype=torch.float32, device=device),
#         indexing="xy",
#     )

#     # Normalize directions for each focal length in the batch
#     dirs_list = []
#     focals = focals.view(batch_size, 1)  # Ensure focals has shape (B, 1)
#     for b in range(batch_size):
#         focal = focals[b].item()
#         dirs = torch.stack(
#             [
#                 (i - ray_W * 0.5 + margin_W) / focal,  # X direction
#                 -(j - ray_H * 0.5 + margin_H) / focal,  # Y direction
#                 -torch.ones_like(i),  # Z direction (negative for camera space)
#             ],
#             dim=-1,
#         )  # Shape (ray_len, ray_len, 3)
#         dirs_list.append(dirs)

#     dirs = torch.stack(dirs_list)  # Shape (B, ray_len, ray_len, 3)

#     # Rotate directions by c2w (camera-to-world matrix)
#     rays_d = torch.einsum("bij,...jk->...ik", c2w[None, :3, :3], dirs[...,None])[...,0]  # Shape (B, ray_len, ray_len, 3)

#     # Broadcast ray origins to match directions
#     rays_o = c2w[..., None, None, :3, 3].expand(rays_d.shape)  # Shape (B, ray_len, ray_len, 3)

#     # Reshape rays to a flat format
#     # rays_o = rays_o.reshape(batch_size, -1, 3)  # (B, ray_len*ray_len, 3)
#     # rays_d = rays_d.reshape(batch_size, -1, 3)  # (B, ray_len*ray_len, 3)

#     return rays_o, rays_d


def pose_to_ray(camera_info, depth, img_size, batch_size=1, num_viewpoints=1):

    R = camera_info["rot"]
    t = camera_info["trans"]
    focals = camera_info["intrinsic"][:,:,0,0]
    batch_size = depth.shape[0]

    device = depth.device

    cam_center = -torch.matmul(R, t)  # (N, 3, 1)
    rot = R * torch.tensor([-1,1,-1])[None, None, None, :].to(device)

    camera_poses = torch.cat((rot, cam_center),dim=-1) 
    homogeneous = torch.tensor([0,0,0,1]).to(R.device).expand(batch_size, num_viewpoints, 1, 4)

    c2w = torch.cat((camera_poses, homogeneous), dim=-2)
    rays_o, rays_d = get_rays(img_size[0], img_size[1], focals, c2w, batch_size, device)

    depths = depth.permute(0,1,3,4,2)

    pts_loc = depths * rays_d.to(device) + rays_o.to(device)

    return pts_loc


def compute_plucker_embed(rays_o, rays_d, B, H, W, c2w):

    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1)[...,None]
    
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    plucker = plucker.permute(0, 1, 4, 2, 3)

    return plucker  # (B, V, 6, H, W)