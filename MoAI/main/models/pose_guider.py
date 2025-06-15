# This code is adapted from below and then modified.
# -----------------------------------------------------------------------------
# Moore-AnimateAnyone
# Apache License, Version 2.0
# Copyright @2023-2024 Moore Threads Technology Co., Ltd.
# https://github.com/MooreThreads/Moore-AnimateAnyone
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin

from .motion_module import zero_module
from .resnet import InflatedConv3d

from main.utils import compute_pca
from torchvision.utils import save_image

class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
        plucker_embedding: str = None,
        plucker_out_channels: Tuple[int] = (16, 32, 64, 64),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        self.plucker_embedding = plucker_embedding

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

        if plucker_embedding is not None:
                        
            self.plucker_conv_in = InflatedConv3d(
                6,
                plucker_out_channels[0],
                kernel_size=3,
                padding=1,
            )
            
            self.plucker_blocks = nn.ModuleList()
            
            for i in range(len(plucker_out_channels) - 1):
                # 3×3 conv to refine
                channel_in = plucker_out_channels[i]
                channel_out = plucker_out_channels[i + 1]
                self.plucker_blocks.append(
                    InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
                )
                self.plucker_blocks.append(
                    InflatedConv3d(
                        channel_in, channel_out, kernel_size=3, padding=1, stride=2
                    )
                )
            
            self.plucker_concat_block = zero_module(
                InflatedConv3d(
                    128 + channel_out,
                    128,
                    kernel_size=1,
                    padding=0,
                )
            )

    def forward(self, conditioning, plucker_embedding=None):
        
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
                
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        
        if self.plucker_embedding is not None:
            plucker_embedding = self.plucker_conv_in(plucker_embedding)
            plucker_embedding = F.silu(plucker_embedding)
            
            for block in self.plucker_blocks:
                plucker_embedding = block(plucker_embedding)
                plucker_embedding = F.silu(plucker_embedding)
                
            embedding = torch.cat((embedding, plucker_embedding), dim=1)
            embedding = self.plucker_concat_block(embedding)
            embedding = F.silu(embedding)
            
        embedding = self.conv_out(embedding)

        return embedding



class PoseGuider_Up(ModelMixin):
    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_embedding_channels: int = 320,   # e.g. 128: the #channels of your low-res embedding
        up_conditioning_channels: int = 2048,        # e.g. 3: the #channels you want in the final high-res map
        single_channel_dim: int = 2048,
        aggregation_setting: str = "concat",
        plucker_embedding: str = None,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
        up_block_channels: Tuple[int] = (1024, 512, 256, 128), # e.g. (2048, 1024, 512, 256, 128)
        plucker_out_channels: Tuple[int] = (16, 32, 64, 64),
        vggt_only: bool = False
    ):
        super().__init__()

        # First, lift from your embedding dim into the highest up-sampling channels
        
        self.aggregation_setting = aggregation_setting
        self.plucker_embedding = plucker_embedding
        self.vggt_only = vggt_only
        
        if aggregation_setting == "concat":
            self.one_d_conv = InflatedConv3d(
                up_conditioning_channels,
                out_channels=single_channel_dim,
                kernel_size=1,
                padding=1,
            )
        
        self.up_conv_in = InflatedConv3d(
            single_channel_dim,
            up_block_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Build a sequence of: [conv 3×3] → [deconv 4×4 ↑2] blocks
        self.up_blocks = nn.ModuleList()
        self.transpose_blocks = nn.ModuleList()
        
        for in_ch, out_ch in zip(up_block_channels[:-1], up_block_channels[1:]):
            # 3×3 conv to refine
            self.up_blocks.append(
                InflatedConv3d(in_ch, in_ch, kernel_size=3, padding=1)
            )
            # 4×4 transposed conv to double spatial size and reduce channels
            self.transpose_blocks.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            
        if not vggt_only:
            self.conv_in = InflatedConv3d(
                conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
            )

            self.blocks = nn.ModuleList([])

            for i in range(len(block_out_channels) - 1):
                channel_in = block_out_channels[i]
                channel_out = block_out_channels[i + 1]
                self.blocks.append(
                    InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
                )
                self.blocks.append(
                    InflatedConv3d(
                        channel_in, channel_out, kernel_size=3, padding=1, stride=2
                    )
                )
            
        # Final 3×3 to map into your desired conditioning channels
        self.conv_out = zero_module(
            InflatedConv3d(
                up_block_channels[-1] * 2 if not vggt_only else up_block_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )
        
        if plucker_embedding is not None:
            self.plucker_conv_in = InflatedConv3d(
                6,
                plucker_out_channels[0],
                kernel_size=3,
                padding=1,
            )
            
            self.plucker_blocks = nn.ModuleList()
            
            for i in range(len(plucker_out_channels) - 1):
                # 3×3 conv to refine
                channel_in = plucker_out_channels[i]
                channel_out = plucker_out_channels[i + 1]
                self.plucker_blocks.append(
                    InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
                )
                self.plucker_blocks.append(
                    InflatedConv3d(
                        channel_in, channel_out, kernel_size=3, padding=1, stride=2
                    )
                )
            
            self.plucker_concat_block = zero_module(
                InflatedConv3d(
                    128 + channel_out,
                    128,
                    kernel_size=1,
                    padding=0,
                )
            )
                
            

    def forward(self, geo_embedding=None, vggt_embedding=None, plucker_embedding=None):
        """
        x: (B, conditioning_embedding_channels, D0, H0, W0)
        returns: (B, conditioning_channels, D0*2^N, H0*2^N, W0*2^N)
        """
        if self.aggregation_setting == "concat":
            h = F.silu(self.one_d_conv(vggt_embedding))
            h = F.silu(self.up_conv_in(h))
        
        else:
            h = F.silu(self.up_conv_in(vggt_embedding))
        
        for layer, trans_layer in zip(self.up_blocks, self.transpose_blocks):
            h = layer(h)
            h = trans_layer(h.squeeze(2)).unsqueeze(2)            
            h = F.silu(h)
        
        if not self.vggt_only:
            if geo_embedding is not None:  
                geo_embedding = self.conv_in(geo_embedding)
                geo_embedding = F.silu(geo_embedding)
                        
                for block in self.blocks:
                    geo_embedding = block(geo_embedding)
                    geo_embedding = F.silu(geo_embedding)
                
        h_resized = F.interpolate(h.squeeze(2).float(), size=(64, 64), mode='bilinear', align_corners=False).unsqueeze(2) 

        if self.plucker_embedding is not None:
            plucker_embedding = self.plucker_conv_in(plucker_embedding)
            plucker_embedding = F.silu(plucker_embedding)
            
            for block in self.plucker_blocks:
                plucker_embedding = block(plucker_embedding)
                plucker_embedding = F.silu(plucker_embedding)
            
            if self.plucker_embedding == "vggt_feature":
                h_resized = torch.cat((h_resized, plucker_embedding), dim=1)
                h_resized =  self.plucker_concat_block(h_resized)
                h_resized = F.silu(h_resized)
                
            elif self.plucker_embedding == "geo_feature":
                geo_embedding = torch.cat((geo_embedding, plucker_embedding), dim=1)
                geo_embedding = self.plucker_concat_block(geo_embedding)
                geo_embedding = F.silu(geo_embedding)
        
        if not self.vggt_only:
            full_feat = torch.cat((geo_embedding, h_resized), dim=1)
        else:
            full_feat = h_resized
                            
        return self.conv_out(full_feat) 