from typing import Dict
from jaxtyping import Float

import numpy as np
import torch
import os
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange
# from splatting import splatting_function

def sph2cart(
    azi: Float[Tensor, 'B'],
    ele: Float[Tensor, 'B'],
    r: Float[Tensor, 'B']
) -> Float[Tensor, 'B 3']:
    # z-up, y-right, x-back
    rcos = r * torch.cos(ele)
    pos_cart = torch.stack([
        rcos * torch.cos(azi),
        rcos * torch.sin(azi),
        r * torch.sin(ele)
    ], dim=1)

    return pos_cart

def get_viewport_matrix(
    width: int,
    height: int,
    batch_size: int=1,
    device: torch.device=None,
) -> Float[Tensor, 'B 4 4']:
    N = torch.tensor(
        [[width/2, 0, 0, width/2],
        [0, height/2, 0, height/2],
        [0, 0, 1/2, 1/2],
        [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device
    )[None].repeat(batch_size, 1, 1)
    return N

def get_projection_matrix(
    fovy: Float[Tensor, 'B'],
    aspect_wh: float,
    near: float,
    far: float
) -> Float[Tensor, 'B 4 4']:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def camera_lookat(
    eye: Float[Tensor, 'B 3'],
    target: Float[Tensor, 'B 3'],
    up: Float[Tensor, 'B 3']
) -> Float[Tensor, 'B 4 4']:
    B = eye.shape[0]
    f = F.normalize(eye - target)
    l = F.normalize(torch.linalg.cross(up, f))
    u = F.normalize(torch.linalg.cross(f, l))

    R = torch.stack((l, u, f), dim=1)  # B 3 3
    M_R = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
    M_R[..., :3, :3] = R

    T = - eye
    M_T = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
    M_T[..., :3, 3] = T

    return (M_R @ M_T).to(dtype=torch.float32)

def focal_length_to_fov(
    focal_length: float,
    censor_length: float = 24.
) -> float:
    return 2 * np.arctan(censor_length / focal_length / 2.)

def forward_warper(
    image: Float[Tensor, 'B C H W'],
    screen,
    pcd,
    mvp_mtx: Float[Tensor, 'B 4 4'],
    viewport_mtx: Float[Tensor, 'B 4 4'],
    alpha: float = 0.5
) -> Dict[str, Tensor]:
    H, W = image.shape[2:4]

    # Projection.
    points_c = pcd @ mvp_mtx.mT
    points_ndc = points_c / points_c[..., 3:4]
    # To screen.
    coords_new = points_ndc @ viewport_mtx.mT

    # Masking invalid pixels.
    invalid = coords_new[..., 2] <= 0
    coords_new[invalid] = -1000000 if coords_new.dtype == torch.float32 else -1e+4

    # Calculate flow and importance for splatting.
    new_z = points_c[..., 2:3]
    flow = coords_new[..., :2] - screen[..., :2]
    ## Importance.
    importance = alpha / new_z
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    ## Rearrange.
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W)

    # Splatting.
    warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
    ## mask is 1 where there is no splat
    mask = (warped == 0.).all(dim=1, keepdim=True).to(image.dtype)
    flow2 = rearrange(coords_new[..., :2], 'b (h w) c -> b c h w', h=H, w=W)

    output = dict(
        warped=warped,
        mask=mask,
        correspondence=flow2
    )

    return output


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


def depth_metrics(pred: torch.Tensor,
                  gt:   torch.Tensor,
                  mask: torch.BoolTensor = None):
    """
    Compute common depth‐prediction metrics between `pred` and `gt`.

    Args:
        pred (B,H,W) or (B,1,H,W): predicted depths
        gt   (B,H,W) or (B,1,H,W): ground‐truth depths
        mask (same shape): optional boolean mask of valid pixels

    Returns:
        dict with keys "AbsRel", "SqRel", "RMSE", "delta1"
    """
    # ensure shape [B,H,W]
    if pred.dim()==4: pred = pred.squeeze(1)
    if gt.dim()==4:   gt   = gt.squeeze(1)

    # valid = finite & positive gt
    valid = torch.isfinite(gt) & (gt>0)
    if mask is not None:
        valid &= mask

    pred, gt = pred[valid], gt[valid]
    N = pred.numel()
    try:
        if N == 0:
            raise ValueError("No valid pixels!")

        diff = pred - gt
        abs_diff = diff.abs()
        sq_diff = diff**2

        # AbsRel
        abs_rel = (abs_diff / gt).mean()

        # SqRel
        sq_rel = (sq_diff / gt).mean()

        # RMSE
        rmse = torch.sqrt(sq_diff.mean())

        # delta < 1.25
        # compute max (pred/gt, gt/pred)
        ratio = torch.max(pred/gt, gt/pred)
        delta1 = (ratio < 1.25).float().mean()

        return {
            "AbsRel": abs_rel.item(),
            "SqRel":  sq_rel.item(),
            "RMSE":   rmse.item(),
            "delta<1.25": delta1.item()
        }
    
    except:
        return None
    

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

    # Reshape depth map to shape (1, 1, H, W) for convolution

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