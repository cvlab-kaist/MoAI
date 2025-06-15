from pathlib import Path
from typing import Callable, List, Type, Any, Dict, Tuple, Union, Optional
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

import copy
import math


class UNetCrossAttentionHooker():
    def __init__(
            self,
            is_train:bool=True,
    ):
        self.cross_attn_maps=[]
        self.is_train=is_train
    
    def clear(self):
        self.cross_attn_maps.clear()

    def _unravel_attn(self, x, n_heads):
        # type: (torch.Tensor, int) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        # n_heads: int
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        maps = []
        x = x.permute(2, 0, 1)

        for map_ in x:
            if not self.is_train:
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
            maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height*width)
        maps=maps.permute(1,0,2) # shape: (heads, tokens, height*width)
        maps=maps.reshape([maps.shape[0]//n_heads,n_heads,*maps.shape[1:]]) # shape: (batch_size, heads, tokens, height*width)
        maps=maps.permute(0,2,1,3) # shape: (batch_size, tokens, heads, height*width)
        return maps

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attn=(encoder_hidden_states is not None)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attn:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            maps = self._unravel_attn(attention_probs, attn.heads)
            self.cross_attn_maps.append(maps)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

class XformersCrossAttentionHooker():
    def __init__(
            self,
            is_train: bool=True,
            num_ref_views: int=2,
            setting: str = "reading",
            cross_attn: bool = False,
            attention_op: Optional[Callable] = None,
            save_attn_map: bool = False
    ):
        self.attention_op = attention_op
        self.image_attention_dict={"key":[], "query":[], "value":[]}
        self.layer_list = []

        self.is_train=is_train
        self.num_ref_views = num_ref_views
        self.setting = setting
        self.cross_attn = cross_attn
                
        self.layer_counter = 0
        self.reading_counter = 0
        
        self.save_attn_map = save_attn_map
        self.cross_attn_maps = {}
        
        self.current_timestep = None
        
        self.y_idxs = torch.tensor([256, 128, 384])
        self.x_idxs = torch.tensor([256, 128, 384])

    def clear(self):
        self.image_attention_dict["key"].clear()
        self.image_attention_dict["query"].clear()
        self.image_attention_dict["value"].clear()
        
        self.layer_list.clear()
        self.layer_counter = 0
        self.reading_counter = 0
    
    def attn_map_clear(self):
        self.cross_attn_maps.clear()

    def _unravel_attn(self, x, n_heads):
        # type: (torch.Tensor, int) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        # n_heads: int
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        maps = []
        x = x.permute(2, 0, 1)

        for map_ in x:
            if not self.is_train:
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
            maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height*width)
        maps=maps.permute(1,0,2) # shape: (heads, tokens, height*width)
        maps=maps.reshape([maps.shape[0]//n_heads,n_heads,*maps.shape[1:]]) # shape: (batch_size, heads, tokens, height*width)
        maps=maps.permute(0,2,1,3) # shape: (batch_size, tokens, heads, height*width)
        return maps


    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)    
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.setting == "reading" and self.layer_counter in self.layer_list:
            query = self.image_attention_dict["query"].pop(0)
            if self.cross_attn == False:
                q_dim = query.shape[1]
                key = self.image_attention_dict["key"].pop(0)[:,:q_dim]
            else:
                key = self.image_attention_dict["key"].pop(0)
                                                            
            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value).contiguous()   
                    
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()        

        if self.setting == "writing":
            if key.shape[-2] == query.shape[-2] * (self.num_ref_views + 1):
                # import pdb; pdb.set_trace()
                self.image_attention_dict["query"].append(query)

                if self.cross_attn == False:
                    q_dim = query.shape[1]
                    self.image_attention_dict["key"].append(key[:,:q_dim])
                else:
                    self.image_attention_dict["key"].append(key)
                                               
                if self.save_attn_map:
                    attention_probs = attn.get_attention_scores(query, key, attention_mask)

                    maps = self._unravel_attn(attention_probs, attn.heads)  
                                      
                    side_dim = int(math.sqrt(maps.shape[-1]))
                    down_scale = 512 // side_dim
                    two, cross_dim, heads, self_dim = maps.shape
                    reshaped_maps = maps.reshape(two, cross_dim, heads, side_dim, side_dim)
                    
                    y_idxs = torch.round(self.y_idxs / down_scale).int()
                    x_idxs = torch.round(self.x_idxs / down_scale).int()
                    
                    attn_maps = reshaped_maps[...,y_idxs,x_idxs]
                                                            
                    self.cross_attn_maps[self.current_timestep].append(attn_maps)
                
                num = self.layer_counter
                self.layer_list.append(num)
                
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        self.layer_counter += 1

        # import pdb; pdb.set_trace()

        return hidden_states
