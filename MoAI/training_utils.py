import sys
# sys.path.append("extern/dust3r")

# from dust3r.inference import loss_of_one_batch, inference
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# # from dust3r.utils.image import load_images
# from dust3r.image_pairs import make_pairs

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from einops import rearrange
import torch.nn.functional as F
# from genwarp.utils.splatting import splatting_function
import numpy as np
import matplotlib.pyplot as plt
import wandb
import importlib
import os
import os.path as osp
import shutil
import sys
from PIL import Image
import PIL
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf

import tqdm
# from dust3r.utils.device import to_cpu, collate_with_cat
# from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
# from dust3r.utils.misc import invalid_to_nans
# from dust3r.utils.geometry import depthmap_to_pts3d, geotrf


# adpated from dust3r inference code
def load_model(model_path, device):
    print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    print(f"instantiating : {args}")
    net = eval(args)
    print(net.load_state_dict(ckpt['model'], strict=False))
    return net.to(device)


def pred_depth_inference(path, model, size=512, device=None):
    images = load_images([path, path], size=size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    intrinsics = scene.get_intrinsics()
    rel_pose = torch.inverse( torch.inverse(poses[:1, ...])@poses[1:, ...] )
    # output dim: (B, ...)
    return output['pred1']['pts3d'][:1, ..., -1].to(device), confidence_masks[0].unsqueeze(0).to(device), rel_pose.to(device), focals[:1].to(device), intrinsics[:1].to(device)


def dust3r_matcher_multiprocessing(batch, model=None, num_workers=6, device=None):
    pass

def forward_warper(images, depths, R, T, K=None, H=512, W=512, device=None): # from GeoFree code

    def pi_inv(K, x, d):
        fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2,
                                           1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
        X_x = d * (x[..., 0] - cx) / fx
        X_y = d * (x[..., 1] - cy) / fy
        X_z = d
        X = torch.stack([X_x, X_y, X_z], dim=-1)
        return X
        
    def x_2d_coords(h, w, device):
        x_2d = torch.zeros((h, w, 2), device=device)
        for y in range(0, h):
            x_2d[y, :, 1] = y
        for x in range(0, w):
            x_2d[:, x, 0] = x
        return x_2d
    
    def transpose(R, t, X):
        b, h, w, c = X.shape
        X = rearrange(X, 'b h w c -> b c (h w)')
        X_after_R = R@X + t[:, :, None]
        X_after_R = rearrange(X_after_R, 'b c (h w) -> b h w c', h=h)
        return X_after_R

    if K is None:
        focal = (5.8269e+02, 5.8269e+02)
        K = torch.tensor([
            [focal[0], 0., W/2],
            [0., focal[1], H/2],
            [0., 0., 1.]], device=device)
        K = K[None, ...]


    if isinstance(depths, np.ndarray):
        depths = torch.tensor(depths).to(device)
    if isinstance(R, np.ndarray):
        R = torch.tensor(R).float().to(device)
        T = torch.tensor(T).float().to(device)

    if R.dim() == 2:
        R = R[None, ...]
        T = T[None, ...]

    if isinstance(images, Image.Image):
        images =  transforms.functional.to_tensor(images).to(device)

    if images.dim() == 3:
        images = images[None, ...]
        B = 1
    else:
        B = images.shape[0]
    
    if depths.dim() == 2:
        depths = depths[None, None, ...]
    elif depths.dim() == 3:
            depths = depths.unsqueeze(1)
    
    
    # unproj. / rotate / translate
    # with torch.autocast(device, enabled=False):
    coords = x_2d_coords(H, W, device=device)[None, ...].repeat(B, 1, 1, 1)
    coords_3d = pi_inv(K, coords, depths[:,0,...])
    coords_world = transpose(R, T, coords_3d)
    coords_world = coords_world.reshape((-1, H, W, 3))
    coords_world = rearrange(coords_world, 'b h w c -> b c (h w)')

     # proj.
    proj_coords_3d =  K[:, :3, :3]@coords_world
    proj_coords_3d = rearrange(proj_coords_3d, 'b c (h w) -> b h w c', h=H, w=W)
    proj_coords = proj_coords_3d[..., :2]/(proj_coords_3d[..., 2:3]+1e-6)

    # masking
    mask = depths[:,0,...] == 0
    proj_coords[mask] = -1000000 if proj_coords.dtype == torch.float32 else -1e+4
    back_mask = proj_coords_3d[..., 2:3] <= 0
    back_mask = back_mask.repeat(1, 1, 1, 2)
    proj_coords[back_mask] = -1000000 if proj_coords.dtype == torch.float32 else -1e+4

    # proj.
    new_z = proj_coords_3d[..., 2:3].permute(0,3,1,2) # B H W 1 -> B 1 H W
    flow = proj_coords - coords
    flow = flow.permute(0,3,1,2) # B H W 2 -> B 2 H W
    
    alpha = 0.5
    importance = alpha/new_z
    importance_min = importance.amin((1,2,3),keepdim=True)
    importance_max = importance.amax((1,2,3),keepdim=True)
    importance=(importance-importance_min)/(importance_max-importance_min+1e-6)*10-10
    importance = importance.exp()
    input_data = torch.cat([importance*images, importance*new_z, importance], 1)
    output_data = splatting_function("summation", input_data, flow)

    
    renderings = output_data[:,:-1,:,:] / (output_data[:,-1:,:,:]+1e-6)
    renderings, warped_depths = renderings[:,:-1,:,:], renderings[:,-1:,:,:]
    masks = (renderings == 0.).all(dim=1).int()
    
    if torch.isnan(renderings).sum() > 0:
        print("NaNs in renderings")
        wandb.alert(title="NaNs in renderings", text="NaNs in renderings")
        renderings = torch.zeros_like(renderings)
        warped_depths = torch.zeros_like(warped_depths)
    
    return renderings, masks.float(), warped_depths, proj_coords
    
        
def plucker_embedding(R, t, K, H, W, device):
    def custom_meshgrid(*args):
        return torch.meshgrid(*args, indexing='ij')

    return_dim = 4
    if R.dim() == 3:
        return_dim = 3
        R = R[:, None, ...]
        K = K[:, None, ...]
        t = t[:, None, ...]  

    # c2w: B, V, 4, 4
    B, V = R.shape[0], R.shape[1]
    c2w = torch.cat([R,t[..., None]], dim=-1)
    lastrow = torch.zeros(B, V, 1, 4).to(device)
    lastrow[..., -1] = 1
    c2w = torch.cat( [c2w,lastrow], dim=-2)
    
    # K: B, V, 3, 3

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    # fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1
    fx, fy, cx, cy = K[..., 0,0, None], K[...,1,1, None], K[...,0,2, None], K[...,1,2, None]

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker if return_dim == 4 else plucker[:, 0, ...] 


def depth_embedding(depth, R, t, K, H, W, embed_func, device):
    def custom_meshgrid(*args):
        return torch.meshgrid(*args, indexing='ij')

    # j: y axis, i: x axis
    j_2d, i_2d = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )

    coords = torch.stack([i_2d[None, ...] / W, j_2d[None, ...] / H], dim=-1)
    out_coords = embed_func(coords)
    R_w = R[:,0,...] if R.dim() == 4 else R
    t_w = t[:,0,...] if t.dim() == 3 else t
    out_coords, masks, _, _ = forward_warper(out_coords.permute(0,3,1,2), depth, R_w, t_w ,K, H=H, W=W, device=device)
        
    return out_coords

   
def camera_controller(key_sequence, speed_scale=1., rotation_scale=1.):
    camera_pos = np.array([0.0, 0.0, 0.0])
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, 1.0, 0.0])
    camera_yaw = 0
    camera_pitch = 0
    
    SPEED = 0.25 * speed_scale
    ROTATE_SENSITIVITY = 10.0 * rotation_scale
    
    def normalize(x):
        return x/np.linalg.norm(x)
    def cosd(x):
        return np.cos(np.deg2rad(x))
    def sind(x):
        return np.sin(np.deg2rad(x))
    def rotate_around_axis(angle, axis):
        axis = normalize(axis)
        rotation = np.array([[cosd(angle)+axis[0]**2*(1-cosd(angle)),
                              axis[0]*axis[1]*(1-cosd(angle))-axis[2]*sind(angle),
                              axis[0]*axis[2]*(1-cosd(angle))+axis[1]*sind(angle)],
                             [axis[1]*axis[0]*(1-cosd(angle))+axis[2]*sind(angle),
                              cosd(angle)+axis[1]**2*(1-cosd(angle)),
                              axis[1]*axis[2]*(1-cosd(angle))-axis[0]*sind(angle)],
                             [axis[2]*axis[0]*(1-cosd(angle))-axis[1]*sind(angle),
                              axis[2]*axis[1]*(1-cosd(angle))+axis[0]*sind(angle),
                              cosd(angle)+axis[2]**2*(1-cosd(angle))]])
        return rotation

    def look_to(pos, dir, up):
      right = normalize(np.cross(up, dir))
      R = np.zeros((4, 4))
      R[0,0:3] = normalize(right)
      R[1,0:3] = normalize(np.cross(dir, right))
      R[2,0:3] = normalize(dir)
      R[3,3] = 1
      trans_matrix = np.array([[1.0, 0.0, 0.0, -camera_pos[0]],
                               [0.0, 1.0, 0.0, -camera_pos[1]],
                               [0.0, 0.0, 1.0, -camera_pos[2]],
                               [0.0, 0.0, 0.0,            1.0]])
      tmp = R@trans_matrix
      return tmp[:3,:3], tmp[:3,3]

    for key in key_sequence:
        if key == 'w': # forward
            camera_pos += SPEED*normalize(camera_dir)
        elif key == 's': # backward
            camera_pos -= SPEED*normalize(camera_dir)
        elif key == 'a': # left 
            camera_pos += SPEED*normalize(np.cross(camera_dir, camera_up))
        elif key == 'd': # right
            camera_pos -= SPEED*normalize(np.cross(camera_dir, camera_up))
        elif key == 'q': # rotate leftside
            camera_yaw += ROTATE_SENSITIVITY
        elif key == 'e': # rotate rightside
            camera_yaw -= ROTATE_SENSITIVITY
        elif key == 'r': # rotate upward
            camera_pitch -= ROTATE_SENSITIVITY
        elif key == 'f': # rotate downward
            camera_pitch += ROTATE_SENSITIVITY

    rotation = np.array([[cosd(-camera_yaw), 0.0, sind(-camera_yaw)],
                             [0.0, 1.0, 0.0],
                             [-sind(-camera_yaw), 0.0, cosd(-camera_yaw)]])
    camera_dir = rotation@camera_dir
    rotation = rotate_around_axis(camera_pitch, np.cross(camera_dir, camera_up))
    camera_dir = rotation@camera_dir
    show_R, show_t = look_to(camera_pos, camera_dir, camera_up) # look from pos in direction dir
    return show_R, show_t[None, ...].T


def camera_controller_game(key_sequence, speed_scale=1., rotation_scale=1.):
    
    R, T = camera_controller(key_sequence[0], speed_scale) if len(key_sequence) > 0 else camera_controller("")
    for key in key_sequence[1:]:
        curr_R, curr_T = camera_controller(key, speed_scale=speed_scale, rotation_scale=rotation_scale)
        R = R @ curr_R
        T = T + curr_T
    
    return R, T


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def get_coords(B,H,W, device, dtype):
    def custom_meshgrid(*args):
        return torch.meshgrid(*args, indexing='ij')
    
    j_2d, i_2d = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
    )
    coords = torch.stack([i_2d[None, ...] / W, j_2d[None, ...] / H], dim=-1)
    coords = coords.repeat(B,1,1,1)
    
    return coords
    


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)
            
# Adapted from DUSt3R
def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if isinstance(path, PIL.Image.Image):
            img = exif_transpose(path)
        elif not path.lower().endswith(supported_images_extensions):
            continue
        else:
            img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)