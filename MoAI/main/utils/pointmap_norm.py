import torch
from torchvision.utils import save_image

class PointmapNormalizer(torch.nn.Module):
    def __init__(self, ptsmap_min, ptsmap_max, k=0.8, device="cpu"):
        """
        Custom normalization function that:
        1. Linearly normalizes values in (a, b) to (-k, k)
        2. Uses logarithmic scaling outside (a, b), constrained to (-1, -k) or (k, 1)

        Args:
            a (float): Lower bound of linear normalization range
            b (float): Upper bound of linear normalization range
            k (float): Scaling factor for the linear range
        """
        super().__init__()

        self.ptsmap_min = ptsmap_min
        self.ptsmap_max = ptsmap_max
        self.border_value = k  # Should be in (0, 1] to ensure asymptotic bounds at -1 and 1

    def exp_func(self, x):
        return 1 - torch.exp((1 - x)/10)

    def forward(self, pts, current_data=None):
        """
        Applies the transformation to input n.

        Args:
            n (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor
        """
        # Ensure k is within the correct range (0, 1]
        assert 0 < self.border_value <= 1, "k should be between (0, 1] to ensure outputs are bounded within (-1,1)"

        shape = (1,) * (pts.ndimension() - 1) + (3,)

        if current_data is None:
            min = self.ptsmap_min.view(shape).expand_as(pts)
            max = self.ptsmap_max.view(shape).expand_as(pts)
        else:
            min = pts.reshape(-1,3).min(dim=0)[0].view(shape).expand_as(pts)
            max = pts.reshape(-1,3).max(dim=0)[0].view(shape).expand_as(pts)

        # linear_mask_dim = (pts.to(min.device) >= min) & (pts.to(min.device) <= max)
        
        # linear_mask = (linear_mask_dim[...,0] * linear_mask_dim[...,1] * linear_mask_dim[...,2])[...,None]
        # log_mask = ~linear_mask
        
        linear_pts = (pts - min) / (max - min)
        linear_pts = linear_pts * (2 * self.border_value) - self.border_value

        # log_pts = (1 - self.border_value) * self.exp_func(torch.abs(linear_pts) + 1 - self.border_value) + self.border_value

        # v = linear_pts / torch.abs(linear_pts)
        
        # log_pts = v * log_pts * log_mask
        # linear_pts = linear_pts * linear_mask

        # final_pts = linear_pts + log_pts
        
        final_pts = linear_pts
        
        return final_pts
