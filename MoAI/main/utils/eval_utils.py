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

from dacite import Config, from_dict
from jaxtyping import Float, Int64

@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]

class EvalBatch():
    def __init__(
            self,
            device   
        ):
        self.chunk_num = 0
        self.example_num = 0
        self.device = device
        self.dtype = torch.bfloat16  # or torch.float16
        
        real_root = "/mnt/data2/minseop/re10k/test_partial"
        self.example_files = sorted(Path(real_root).rglob("*.torch"))

        self.chunk_overview = []

        for file_path in self.example_files:
            examples = torch.load(file_path)
            self.chunk_overview.append(len(examples))
            
        self.crop_transform = tf.Compose([
            tf.CenterCrop(360),
            tf.Resize((512, 512)),   # Resize to a consistent size (adjust as needed)
            tf.ToTensor()             # Convert to tensor and normalize to [0, 1]
        ])
        
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        
        self.crop_transform = tf.Compose([
            tf.Resize(512, interpolation=tf.InterpolationMode.BICUBIC),  # Resize shortest side to 512
            tf.CenterCrop(512),  # Center crop to 512x512
            tf.ToTensor()    
        ])
        
        index_path = Path("/mnt/data1/minseop/multiview-gen/evaluation_index_re10k.json")
        
        dacite_config = Config(cast=[tuple])
        with index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }
        
        
    def convert_images(
        self,
        images,
    ):
        to_tensor = tf.ToTensor()
        torch_images = []
            
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(to_tensor(image))
            
        return torch.stack(torch_images)
    
    
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
        
        
    def realestate_eval(self, num_viewpoints = 2, setting = "extrapolate"):

        self.example_num += 1
        
        if self.chunk_overview[self.chunk_num] == self.example_num:
            self.chunk_num += 1
            self.example_num = 0
            
        example = torch.load(self.example_files[self.chunk_num])[self.example_num]

        key = example["key"]
        num_imgs = len(example["images"])
        
        if setting == "interpolate":
            entry = self.index.get(key)
            loop = True
            while loop:
                if entry is None:
                    self.example_num += 1
                    if self.chunk_overview[self.chunk_num] == self.example_num:
                        self.chunk_num += 1
                        self.example_num = 0
                    example = torch.load(self.example_files[self.chunk_num])[self.example_num]
                    key = example["key"]
                    num_imgs = len(example["images"])
                    entry = self.index.get(key)
                    
                else:
                    loop = False
                    context_indices = torch.tensor(entry.context, dtype=torch.int64, device=self.device)
                    target_indices = torch.tensor(entry.target, dtype=torch.int64, device=self.device)
        
        elif setting == "extrapolate":
            entry = self.index.get(key)
            loop = True

            while loop:
                if entry is None:
                    self.example_num += 1
                    if self.chunk_overview[self.chunk_num] == self.example_num:
                        self.chunk_num += 1
                        self.example_num = 0
                    example = torch.load(self.example_files[self.chunk_num])[self.example_num]
                    key = example["key"]
                    num_imgs = len(example["images"])
                    entry = self.index.get(key)
                
                else:
                    loop = False
                    context_indices = torch.tensor([int(num_imgs * 2/3), int(num_imgs * 4/5), int(num_imgs * 1/2), int(num_imgs * 3/5)])
                    target_indices = torch.tensor([int(num_imgs * 1/3), int(num_imgs * 2/5), int(num_imgs * 1/4)])
        else:
            raise ValueError("Invalid setting. Choose 'interpolate' or 'extrapolate'.")

        idx = torch.cat((context_indices, target_indices)).tolist()
        
        imagelist = [example['images'][i] for i in idx]
                
        images = self.convert_images(imagelist)
        images_vggt = F.interpolate(images, size=(294, 518), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
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
            
            # import pdb; pdb.set_trace()
            
            #jiho: hardcoding: used_idx selected in train_vggt_inference.py, ctx= 0,1 , tgt = 4,5,6
            if setting == "extrapolate":
                used_idx = [idx[0], idx[1], idx[4], idx[5], idx[6]]
            elif setting == "interpolate":
                used_idx = [idx[0], idx[1], idx[2], idx[3], idx[4]]
            name = key + '_idx_' + '_'.join(map(str, used_idx))
            output = dict(image = images[None,...].to(self.device), points = pts[None,...].to(self.device), intrinsic = intrinsic.to(self.device), extrinsic = extrinsic.to(self.device), conf=None, dataset=[1], 
                          instance_name = name)
        
        return output

    def dtu_eval(self, num_src = 2, num_target = 1, setting = "extrapolate"): # extrapolating settings
        idxpair_per_scene = 4 # mvsplat idx
        from einops import repeat, rearrange # should move to top
        ### # Should move to init ####
        root ="/mnt/data1/jiho/datasets/dtu_mvsplat/test"
        self.example_files = sorted(Path(root).rglob("*.torch"))
        self.chunk_overview = []
        for file_path in self.example_files:
            examples = torch.load(file_path)
            self.chunk_overview.append(len(examples))
        ##############################

        
        example = torch.load(self.example_files[self.chunk_num])[self.example_num]
        if not hasattr(self, "idxpair_step"):
            self.idxpair_step = -1
            
        
        self.idxpair_step += 1
        
        if self.idxpair_step == idxpair_per_scene:
            self.idxpair_step = 0
            self.example_num += 1
            if self.chunk_overview[self.chunk_num] == self.example_num:
                self.chunk_num += 1
                self.example_num = 0
            
        key = example["key"]
        # Index Selection
        
        # 1) heuristic
        # num_imgs = len(example["images"])
        # context_indices = torch.tensor([int(num_imgs * 2/3), int(num_imgs * 4/5)])
        # target_indices = torch.tensor([int(num_imgs * 1/3), int(num_imgs * 2/5), int(num_imgs * 1/4)])
        # idx = torch.cat((context_indices, target_indices)).tolist()
        
        # 2) mvsplat processed
        idx_path= '/mnt/data1/jiho/datasets/dtu_mvsplat/assets'
        idx_path = os.path.join(idx_path, f"nctx{num_src + num_target - 1}.json") # nctx 1 ~ 9, default_target_num = 1
        with open(idx_path, "r") as f:
            idx_full = json.load(f)
        idx_example_keys = [k  for k in idx_full.keys() if k.startswith(key)]
        # idx_example_key = np.random.choice(idx_example_keys).item()
        idx_example_key = idx_example_keys[self.idxpair_step]
        idx_selected = idx_full[idx_example_key]
        idx_selected = idx_selected['context'] + idx_selected['target'] # merge, we will select target from below
        
        if setting == "interpolate":
            idx = idx_selected
        elif setting == "extrapolate":
            # target selection: fartherest camera
            def convert_poses(poses):
                b, _ = poses.shape
                # Convert the intrinsics to a 3x3 normalized K matrix.
                intrinsics = torch.eye(3, dtype=torch.float32)
                intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
                fx, fy, cx, cy = poses[:, :4].T
                intrinsics[:, 0, 0] = fx
                intrinsics[:, 1, 1] = fy
                intrinsics[:, 0, 2] = cx
                intrinsics[:, 1, 2] = cy

                # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
                w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
                w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
                return w2c.inverse(), intrinsics
            extri, _ = convert_poses(example["cameras"])
            extri = extri[idx_selected]
            T = extri[:, :3, -1]
            diff = T.unsqueeze(1) - T.unsqueeze(0)  # Shape: (N, N, 3)
            dist = torch.norm(diff, dim=2).mean(dim=1) 
            idx_target = torch.topk(dist, num_target).indices.tolist()
            idx_target = [idx_selected[i] for i in idx_target]
            idx_source = list(set(idx_selected) - set(idx_target))            
            idx = idx_source + idx_target
        else:
            raise ValueError("Invalid setting. Choose 'interpolate' or 'extrapolate'.")

        imagelist = [example['images'][i] for i in idx]
                
        images = self.convert_images(imagelist)
        images_vggt = F.interpolate(images, size=(294, 518), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
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
            
            name = idx_example_key + '_idx_' + '_'.join(map(str, idx))
            output = dict(image = images[None,...].to(self.device), points = pts[None,...].to(self.device), intrinsic = intrinsic.to(self.device), extrinsic = extrinsic.to(self.device), conf=None, dataset=[1],
                          instance_name = name)
            print("name : ", name)
        return output

def revisit_eval(scene_dir):
    
    batch_dir = os.path.join(scene_dir, "batch_info.pt")
    info_dir =  os.path.join(scene_dir, "run_info.json")
    
    with open(info_dir, "r") as f:
        info_data = json.load(f)

    src_idx = info_data["source_idx"]
    target_idx_list = info_data["target_idx"]
        
    batch = torch.load(batch_dir)
    
    keypoints_dict = {}
    
    for current_path, dirs, _ in os.walk(scene_dir):
        dirs = sorted(dirs)
        for d in dirs:
            tgt_idx = d.split("_")[-1]  
            keypoint_dir = os.path.join(scene_dir, d, "keypoints.json")
            
            with open(keypoint_dir, "r") as f:
                keypoint_info = json.load(f)
            
            keypoints_dict[tgt_idx] = keypoint_info["keypoints"]
            
    
    return batch, src_idx, target_idx_list, keypoints_dict


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