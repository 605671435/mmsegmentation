# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils.ex_attention import EX_Module

class ExAttention(EX_Module):
    def __init__(self, dim, **kwargs):
        super(ExAttention, self).__init__(in_channels=dim, channels=dim, **kwargs)

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act1_layer=dict(type='GELU'),
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=7,
                 padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = build_activation_layer(act1_layer)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding module implemented by a layer of convolution.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    Args:
        patch_size (int): Patch size of the patch embedding. Defaults to 16.
        stride (int): Stride of the patch embedding. Defaults to 16.
        padding (int): Padding of the patch embedding. Defaults to 0.
        in_chans (int): Input channels. Defaults to 3.
        embed_dim (int): Output dimension of the patch embedding.
            Defaults to 768.
        norm_layer (module): Normalization module. Defaults to None (not use).
    """

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MetaFormerBlock(BaseModule):
    def __init__(self,
                 dim,
                 token_mixer=nn.Identity,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=nn.GELU,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()


    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
                self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def basic_blocks(dim,
                 index,
                 layers,
                 token_mixer=nn.Identity,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=.0,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(MetaFormerBlock(
            dim,
            token_mixer=token_mixer,
            mlp_ratio=mlp_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            drop=drop_rate,
            drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks

@MODELS.register_module()
class MetaFormer(BaseModule):
    def __init__(self,
                 layers,
                 embed_dims=None,
                 token_mixers=None,
                 mlp_ratios=[4, 4, 4, 4],
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 add_pos_embs=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 init_cfg=None,
                 **kwargs):

        super().__init__(init_cfg=init_cfg)

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0])

        if add_pos_embs is None:
            add_pos_embs = [None] * len(layers)
        if token_mixers is None:
            token_mixers = [nn.Identity] * len(layers)
        # set the main block in network
        network = []
        for i in range(len(layers)):
            if add_pos_embs[i] is not None:
                network.append(add_pos_embs[i](embed_dims[i]))
            stage = basic_blocks(embed_dims[i],
                                 i,
                                 layers,
                                 token_mixer=token_mixers[i],
                                 mlp_ratio=mlp_ratios[i],
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1]))

        self.network = nn.ModuleList(network)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 7 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        if self.out_indices:
            for i_layer in self.out_indices:
                layer = build_norm_layer(norm_cfg,
                                         embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            # Include both block and downsample layer.
            module = self.network[i]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(MetaFormer, self).train(mode)
        self._freeze_stages()

@MODELS.register_module()
class CAFormer_s18(MetaFormer):
    def __init__(self, **kwargs):
        super(CAFormer_s18, self).__init__(
            layers=[3, 3, 9, 3],
            embed_dims=[64, 128, 320, 512],
            token_mixers=[SepConv, SepConv, ExAttention, ExAttention],
            **kwargs)