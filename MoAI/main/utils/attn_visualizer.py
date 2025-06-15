import torch
import numpy as np
import torchvision

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2

from PIL import Image

def stitch_side_by_side(images, padding=0, bg_color=(0,0,0)):
    """
    Stitch a list of PIL Images horizontally.

    Args:
        images (List[PIL.Image]): exactly four images (or any number).
        padding (int): pixels of space between images.
        bg_color (tuple): background color RGB for the canvas.

    Returns:
        PIL.Image: the stitched panorama.
    """
    # Compute the size of the final canvas
    widths, heights = 512, 512
    total_width = widths * len(images) + padding * (len(images) - 1)
    max_height  = heights

    # Create the empty canvas
    canvas = Image.new('RGB', (total_width, max_height), color=bg_color)

    # Paste each image next to the previous
    x_offset = 0
    for img in images:
        # If the img is shorter than max_height, you can vertically center it:
        y_offset = (max_height - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))
        x_offset += img.width + padding

    return canvas

def get_attn_map(attn_layer, background):
    resize = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    # 1t1: target_image_clone / 1t2: source_iamge_clone

    attn_layer = resize(attn_layer.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()
    normalizer = mpl.colors.Normalize(vmin=attn_layer.min(), vmax=attn_layer.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    colormapped_im = (mapper.to_rgba(attn_layer[:,:,0])[:, :, :3] * 255).astype(np.uint8)
    attn_map = cv2.addWeighted(background.copy(), 0.3, colormapped_im, 0.7, 0)
    
    # Find max attention location
    max_idx = np.unravel_index(np.argmax(attn_layer, axis=None), attn_layer.shape)

    # Scale max position to resized image dimensions
    scale_x = 512 / attn_layer.shape[1]  # 512 / 14
    scale_y = 512 / attn_layer.shape[0]  # 512 / 14
    max_x = int(max_idx[1] * scale_x)  # Scale width index
    max_y = int(max_idx[0] * scale_y)  # Scale height index

    # Draw dot on max attention point
    attn_map = cv2.circle(attn_map, (max_x, max_y), 10, (255, 255, 255), -1)  # Blue dot
    attn_map = cv2.circle(attn_map, (max_x, max_y), 6, (255, 0, 0), -1)
    
    return attn_map


def sample_zero_mask_pixels(mask: torch.Tensor, N: int, 
                            grid_size: int = 512, square_size: int = 400) -> torch.Tensor:
    """
    Randomly sample N pixel coordinates (y, x) from a grid of size (grid_size, grid_size),
    but only within the centered square of size (square_size, square_size), 
    and only at locations where mask == 0.

    Args:
        mask (torch.Tensor): Binary mask of shape (grid_size, grid_size), dtype torch.uint8 or bool.
        N (int): Number of samples to draw.
        grid_size (int): Size of the full grid (default 512).
        square_size (int): Size of the centered square (default 400).

    Returns:
        coords (torch.LongTensor): Tensor of shape (N, 2), each row = (y, x).
    """
    assert mask.shape == (grid_size, grid_size), \
        f"Expected mask of shape ({grid_size},{grid_size}), got {tuple(mask.shape)}"
    # compute square bounds
    half = square_size // 2
    center = grid_size // 2
    y0 = center - half
    x0 = center - half
    # crop mask to the square
    sub = mask[y0 : y0 + square_size, x0 : x0 + square_size]
    # find all zero‐locations in the crop
    zero_locs = torch.nonzero(sub == 0, as_tuple=False)  # shape (M, 2), rows = [y_rel, x_rel]
    M = zero_locs.size(0)
    if M == 0 or M < N:
        zero_locs = torch.nonzero(sub == 1, as_tuple=False)
        M = zero_locs.size(0)
        
    # choose N random indices without replacement
    perm = torch.randperm(M, device=zero_locs.device)[:N]
    chosen = zero_locs[perm]
    # shift back to full‐grid coords
    chosen[:, 0] += y0
    chosen[:, 1] += x0
    return chosen  # LongTensor of shape (N,2)

def visualize_attention(ref_images, tgt_img, warp_mask, attn):
    
    pass
