import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import kaiming_init

from mmseg.registry.registry import MODELS
import torch.nn.functional as F

class SelectiveKernelAttn(nn.Module):
    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        # https://github.com/rwightman/pytorch-image-models/blob/709d5e0d9d2d3f501531506eda96a435737223a3/timm/layers/selective_kernel.py
        """ Selective Kernel Attention Module
        Selective Kernel attention mechanism factored out into its own module.
        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.fc_reduce, mode='fan_in')
        kaiming_init(self.fc_select, mode='fan_in')
        self.fc_reduce.inited = True
        self.fc_select.inited = True

    def forward(self, x):
        # [B, 2, C, H, W]
        assert x.shape[1] == self.num_paths

        # [B, C, 1, 1]
        x = x.sum(1).mean((2, 3), keepdim=True)

        # [B, IC, 1, 1]
        x = self.fc_reduce(x)
        # [B, IC, 1, 1]
        x = self.bn(x)
        # [B, IC, 1, 1]
        x = self.act(x)
        # [B, C * 2, 1, 1]
        x = self.fc_select(x)

        B, C, H, W = x.shape
        # [B, 2, C / 2, 1, 1]
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        # [B, 2, C / 2, 1, 1]
        x = torch.softmax(x, dim=1)
        return x

class SK_Module(nn.Module):
    def __init__(self, in_channels, conv_cfg=None):
        super(SK_Module, self).__init__()
        if conv_cfg is not None and conv_cfg['type'] == 'Conv3d':
            self.gap = nn.AdaptiveAvgPool3d(1)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg),
            ConvModule(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg))
        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        u = self.gap(x1 + x2)
        u = self.bottleneck(u)
        softmax_a = self.softmax(u)
        out = x1 * softmax_a + x2 * (1 - softmax_a)
        return out

@MODELS.register_module()
class EX_Module_legacy(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_legacy, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 2
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta
        self.sk = SK_Module(in_channels=in_channels,
                            conv_cfg=conv_cfg)
        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, c // 2, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1)))   #b, c/2, 1
        spatial_attn = self.conv_up(spatial_attn).reshape(b, c, 1) #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, c // 2), dim=1) #b, c/2
        avg_x = avg_x.reshape(b, 1, c // 2)
        theta_x = self.conv_v_left(x).reshape(b, c // 2, h * w)  #b, c/2, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context) #b, 1, h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, h, w)

        parallel = (channel_attn + spatial_attn).reshape(b, c, h, w)
        # sequence = (x * channel_attn) * spatial_attn
        #
        # parallel = x * channel_attn + x * spatial_attn
        sk_attn = self.sk(sequence, parallel)
        # sk_attn = torch.cat((sequence, parallel), dim=1)

        if self.with_self is True:
            self_attn = torch.softmax(sk_attn.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
            value = self.resConv(x)
            self_attn = self_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * sk_attn
        return out + x

@MODELS.register_module()
class EX_Module(nn.Module):
    def __init__(self,
                 in_channels,
                 attn_types=('ch', 'sp'),
                 fusion_type='dsa',
                 ratio=2):
        super().__init__()

        assert isinstance(attn_types, (list, tuple))
        valid_attn_types = ['ch', 'sp']
        assert all([a in valid_attn_types for a in attn_types])
        assert len(attn_types) > 0, 'at least one attention should be used'

        assert isinstance(fusion_type, str)
        valid_fusion_types = ['sq', 'pr', 'dsa', 'None']
        assert fusion_type in valid_fusion_types
        if fusion_type == 'None':
            assert len(attn_types) == 1, 'None fusion only need one attention type.'
        else:
            assert len(attn_types) == 2, 'Fusion need two attention types.'

        self.in_channels = in_channels
        self.channels = in_channels // ratio

        self.attn_types = attn_types
        self.fusion_types = fusion_type

        if 'ch' in attn_types:
            self.conv_q_right = nn.Conv2d(self.in_channels, 1, kernel_size=1)
            self.conv_v_right = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
            self.conv_up = nn.Sequential(
                nn.Conv2d(self.channels, self.in_channels,  kernel_size=1),
                nn.LayerNorm(normalized_shape=[self.in_channels, 1, 1]),
                nn.Sigmoid())
            self.softmax_right = nn.Softmax(dim=2)

        if 'sp' in attn_types:
            self.conv_q_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
            self.conv_v_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
            self.gap = nn.AdaptiveMaxPool2d(output_size=1)
            self.softmax_left = nn.Softmax(dim=1)
            self.softmax_sp = nn.Softmax(dim=2)

        if fusion_type == 'dsa':
            self.sk = SelectiveKernelAttn(channels=self.in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if 'ch' in self.attn_types:
            kaiming_init(self.conv_q_right, mode='fan_in')
            kaiming_init(self.conv_v_right, mode='fan_in')
            self.conv_q_right.inited = True
            self.conv_v_right.inited = True
        if 'sp' in self.attn_types:
            kaiming_init(self.conv_q_left, mode='fan_in')
            kaiming_init(self.conv_v_left, mode='fan_in')
            self.conv_q_left.inited = True
            self.conv_v_left.inited = True

    def forward(self, x):
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        if 'ch' in self.attn_types:
            # Spatial Pooling (psa)
            # [B, IC, H, W]
            input_x = self.conv_v_right(x)
            # [B, IC, H * W]
            input_x = input_x.reshape(b, self.channels, h * w)
            # [B, 1, H, W]
            context_mask = self.conv_q_right(x)
            # [B, 1, H * W]
            context_mask = self.softmax_right(context_mask.reshape(b, 1, h * w))
            # [B, H * W, 1]
            context_mask = context_mask.transpose(1, 2)
            # [B, IC, 1, 1]
            context = torch.matmul(input_x, context_mask).reshape(b, self.channels, 1, 1)
            # [B, C, 1, 1]
            channel_attn = self.conv_up(context)

            if 'sp' not in self.attn_types:
                out = x * channel_attn
                return out + x

            # [B, C, 1]
            channel_attn = channel_attn.reshape(b, c, 1)

        if 'sp' in self.attn_types:
            # Channel Pooling (psa)
            # [B, IC, H, W]
            g_x = self.conv_q_left(x)
            # [B, IC, 1]
            avg_x = self.softmax_left(self.gap(g_x))
            # [B, 1, IC]
            avg_x = avg_x.reshape(b, 1, self.channels)
            # [B, IC, H * W]
            theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)
            # [B, 1, H * W]
            context = torch.matmul(avg_x, theta_x)
            # [B, 1, H * W]
            # spatial_attn = torch.sigmoid(context)
            # [B, 1, H * W]
            spatial_attn = self.softmax_sp(context)

            if 'ch' not in self.attn_types:
                spatial_attn = spatial_attn.reshape(b, 1, h, w)
                out = x * spatial_attn
                return out + x

        if self.fusion_types == 'sq' or self.fusion_types == 'dsa':
            # [B, C, H * W]
            sequence = torch.bmm(channel_attn, spatial_attn)
            # sequence = spatial_attn * channel_attn
            # [B, C, H, W]
            sequence = sequence.reshape(b, c, h, w)
            if 'pr' not in self.fusion_types:
                out = x * sequence
                return out + x
        if self.fusion_types == 'pr' or self.fusion_types == 'dsa':
            # [B, C, H * W]
            parallel = spatial_attn + channel_attn
            # [B, C, H, W]
            parallel = parallel.reshape(b, c, h, w)
            if 'sq' not in self.fusion_types:
                out = x * parallel
                return out + x
            else:
                # [B, 2, C, H, W]
                stack = torch.stack([sequence, parallel], dim=1)
                # [B, 2, C, 1, 1]
                sk_attn = self.sk(stack)
                # [B, 2, C, H, W]
                sq_pr = stack * sk_attn
                # [B, C, H, W]
                sq_pr = torch.sum(sq_pr, dim=1)

                out = x * sq_pr

                return out + x

@MODELS.register_module()
class EX_Module_sp(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio=2):
        super(EX_Module_sp, self).__init__()

        self.in_channels = in_channels
        self.channels = in_channels // ratio

        self.conv_q_right = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        self.conv_v_right = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.channels, self.in_channels,  kernel_size=1),
            nn.LayerNorm(normalized_shape=[self.in_channels, 1, 1]),
            nn.Sigmoid())
        self.softmax_right = nn.Softmax(dim=2)

        self.conv_q_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.conv_v_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.gap = nn.AdaptiveMaxPool2d(output_size=1)
        self.softmax_left = nn.Softmax(dim=1)
        self.softmax_sp = nn.Softmax(dim=2)
        self.sk = SelectiveKernelAttn(channels=self.in_channels)

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def forward(self, x):
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        # Spatial Pooling (psa)
        # [B, IC, H, W]
        input_x = self.conv_v_right(x)
        # [B, IC, H * W]
        input_x = input_x.reshape(b, self.channels, h * w)
        # [B, 1, H, W]
        context_mask = self.conv_q_right(x)
        # [B, 1, H * W]
        context_mask = self.softmax_right(context_mask.reshape(b, 1, h * w))
        # [B, H * W, 1]
        context_mask = context_mask.transpose(1, 2)
        # [B, IC, 1, 1]
        context = torch.matmul(input_x, context_mask).reshape(b, self.channels, 1, 1)
        # [B, C, 1,]
        channel_attn = self.conv_up(context).reshape(b, c, 1)

        # Channel Pooling (psa)
        # [B, IC, H, W]
        g_x = self.conv_q_left(x)
        # [B, IC, 1]
        avg_x = self.softmax_left(self.gap(g_x))
        # [B, 1, IC]
        avg_x = avg_x.reshape(b, 1, self.channels)
        # [B, IC, H * W]
        theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)
        # [B, 1, H * W]
        context = torch.matmul(avg_x, theta_x)
        # [B, 1, H * W]
        # spatial_attn = torch.sigmoid(context)
        # [B, 1, H * W]
        spatial_attn = self.softmax_sp(context)

        # [B, C, H * W]
        sequence = torch.bmm(channel_attn, spatial_attn)
        # sequence = spatial_attn * channel_attn
        # [B, C, H, W]
        sequence = sequence.reshape(b, c, h, w)
        # [B, C, H * W]
        parallel = spatial_attn + channel_attn
        # [B, C, H, W]
        parallel = parallel.reshape(b, c, h, w)

        # [B, 2, C, H, W]
        stack = torch.stack([sequence, parallel], dim=1)
        # [B, 2, C, 1, 1]
        sk_attn = self.sk(stack)
        # [B, 2, C, H, W]
        sq_pr = stack * sk_attn
        # [B, C, H, W]
        sq_pr = torch.sum(sq_pr, dim=1)

        out = x * sq_pr

        return out + x

@MODELS.register_module()
class EX_Module_sp_legacy(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio=2,
                 with_self=True,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_sp_legacy , self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, self.channels, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1

        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1

        spatial_attn = self.conv_up(spatial_attn) #b, c, 1

        if self.with_self is True:
            value = self.resConv(x)
            self_attn = spatial_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * spatial_attn
        return out + x

@MODELS.register_module()
class EX_Module_ch_legacy(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 ratio=1,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_ch_legacy , self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/r, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels), dim=1) #b, c/r
        avg_x = avg_x.reshape(b, 1, self.channels)
        theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)  #b, c/r, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context).reshape(b, 1, h, w)  #b, 1, h, w

        if self.with_self is True:
            value = self.resConv(x)
            self_attn = channel_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * channel_attn
        return out + x

@MODELS.register_module()
class EX_Module_sq_legacy(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 ratio=2,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_sq_legacy, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, self.channels, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1
        spatial_attn = self.conv_up(spatial_attn).reshape(b, c, 1) #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels), dim=1) #b, c/2
        # avg_x = F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels)  # b, c/2
        avg_x = avg_x.reshape(b, 1, self.channels)
        theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)  #b, c/2, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context) #b, 1, h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, h, w)

        if self.with_self is True:
            self_attn = torch.softmax(sequence.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
            value = self.resConv(x)
            self_attn = self_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * sequence
        return out + x

@MODELS.register_module()
class EX_Module_pr_legacy(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 ratio=2,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_pr_legacy, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, self.channels, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1
        spatial_attn = self.conv_up(spatial_attn).reshape(b, c, 1) #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels), dim=1) #b, c/2
        # avg_x = F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels)  # b, c/2
        avg_x = avg_x.reshape(b, 1, self.channels)
        theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)  #b, c/2, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context) #b, 1, h*w

        parallel = (channel_attn + spatial_attn).reshape(b, c, h, w)

        if self.with_self is True:
            self_attn = torch.softmax(parallel.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
            value = self.resConv(x)
            self_attn = self_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * parallel
        return out + x

@MODELS.register_module()
class EX_Module_3D(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 # channels,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_3D, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 2
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta
        self.sk = SK_Module(in_channels=in_channels,
                            conv_cfg=conv_cfg)
        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, d, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, d, h, w
        context_mask = self.conv_q_right(x) #b, 1, d, h, w
        context_mask = F.softmax(context_mask.reshape(b, 1, d * h * w), dim=2) #b, 1, d*h*w
        context_mask = context_mask.reshape(b, 1, d * h * w)
        context = torch.matmul(input_x.reshape(b, c // 2, d * h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1, 1), normalized_shape=(c//2, 1, 1, 1))).reshape(b, c, 1)   #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, d, h, w
        avg_x = F.softmax(F.adaptive_avg_pool3d(g_x, output_size=1).reshape(b, c // 2), dim=1) #b, c/2
        avg_x = avg_x.reshape(b, 1, c // 2)
        theta_x = self.conv_v_left(x).reshape(b, c // 2, d * h * w)  #b, c/2, d*h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, d*h*w
        channel_attn = torch.sigmoid(context) #b, 1, d*h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, d, h, w)

        parallel = F.softmax((channel_attn + spatial_attn).reshape(b, c, d * h * w), dim=2).reshape(b, c, d, h, w)
        # sequence = (x * channel_attn) * spatial_attn
        #
        # parallel = x * channel_attn + x * spatial_attn
        sk_attn = self.sk(sequence, parallel)
        # sk_attn = torch.cat((sequence, parallel), dim=1)

        if self.with_self is True:
            self_attn = sk_attn.reshape(b, c, d * h * w).sum(dim=2).reshape(b, c, 1, 1, 1) + self.resConv(x)
            out = self_attn
        else:
            out = x * sk_attn
        return out
