import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_norm_layer
from ..utils import nlc_to_nchw, nchw_to_nlc
from mmengine.model import ModuleList
from mmcv.cnn.bricks import HSigmoid
class SK_Module(nn.Module):
    def __init__(self, in_channels):
        super(SK_Module, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        bottleneck = ModuleList()
        self.bottleneck = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                bias=False),
            Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
                bias=False))
        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        u = self.gap(x1 + x2)
        u = self.bottleneck(u)
        softmax_a = self.softmax(u)
        out = x1 * softmax_a + x2 * (1 - softmax_a)
        return out


class EX_Module(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def sequence_attention(self, spatial_attn, channel_attn):
        #sequence attention:a*c*s
        b, c, h, w = self.size
        spatial_attn = spatial_attn.reshape(b, 1, h * w)  # 1,h*w
        channel_attn = channel_attn.reshape(b, c, 1)  # c,1
        seq_attn = self.sigmoid(torch.bmm(channel_attn, spatial_attn))  # c,h,w
        seq_attn = seq_attn.reshape(b, c, h, w)
        # seq_attn = self.sigmoid(x * seq_attn)

        return seq_attn

    def selected_attention(self, x, seq_attn, par_attn, hw_shape):
        # select attention
        sk_results = self.sk_module(seq_attn, par_attn)
        sk_results = nchw_to_nlc(sk_results)
        sk_results = nn.Softmax(dim=1)(sk_results)
        sk_results = nlc_to_nchw(sk_results, hw_shape)
        selected_attn = x * sk_results  # c,h,w

        return selected_attn

    def self_attention(self, x, selected_attn):
        # self attention
        b, c, h, w = self.size
        self_attn = selected_attn.reshape(b, c, h * w)  # b*c*n
        self_attn = self_attn.sum(dim=2).reshape(b, c, 1, 1)  # b*c*1*1

        self_attn_res = self.resConv(x)

        self_attn = self_attn + self_attn_res

        return self_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # parallel attention:a*c+a*s
        par_attn = self.sigmoid(spatial_attn + channel_attn)
        # par_attn = self.sigmoid(x * par_attn)

        # sequence attention:a*c*s
        seq_attn = self.sequence_attention(spatial_attn, channel_attn)

        # select attention
        selected_attn = self.selected_attention(x, seq_attn, par_attn, hw_shape)

        # self-attention
        # self_attn = self.softmax(sk_results)
        # self_attn = x * self_attn  # c,h,w
        self_attn = self.self_attention(x, selected_attn)

        # add attention
        out = selected_attn + self_attn

        # out = self.conv2(out)
        return out

class EX_Module_noself(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module_noself, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.sk_module = SK_Module(in_channels=in_channels)

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def sequence_attention(self, spatial_attn, channel_attn):
        #sequence attention:a*c*s
        b, c, h, w = self.size
        spatial_attn = spatial_attn.reshape(b, 1, h * w)  # 1,h*w
        channel_attn = channel_attn.reshape(b, c, 1)  # c,1
        seq_attn = self.sigmoid(torch.bmm(channel_attn, spatial_attn))  # c,h,w
        seq_attn = seq_attn.reshape(b, c, h, w)
        # seq_attn = self.sigmoid(x * seq_attn)

        return seq_attn

    def selected_attention(self, x, seq_attn, par_attn, hw_shape):
        # select attention
        sk_results = self.sk_module(seq_attn, par_attn)
        sk_results = nchw_to_nlc(sk_results)
        sk_results = nn.Softmax(dim=1)(sk_results)
        sk_results = nlc_to_nchw(sk_results, hw_shape)
        selected_attn = x * sk_results  # c,h,w

        return selected_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # parallel attention:a*c+a*s
        par_attn = self.sigmoid(spatial_attn + channel_attn)
        # par_attn = self.sigmoid(x * par_attn)

        # sequence attention:a*c*s
        seq_attn = self.sequence_attention(spatial_attn, channel_attn)

        # select attention
        selected_attn = self.selected_attention(x, seq_attn, par_attn, hw_shape)

        # add attention
        out = selected_attn

        # out = self.conv2(out)
        return out

class EX_Module_noselect_seq(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module_noselect_seq, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        # self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def sequence_attention(self, spatial_attn, channel_attn):
        #sequence attention:a*c*s
        b, c, h, w = self.size
        spatial_attn = spatial_attn.reshape(b, 1, h * w)  # 1,h*w
        channel_attn = channel_attn.reshape(b, c, 1)  # c,1
        seq_attn = self.sigmoid(torch.bmm(channel_attn, spatial_attn))  # c,h,w
        seq_attn = seq_attn.reshape(b, c, h, w)
        # seq_attn = self.sigmoid(x * seq_attn)

        return seq_attn

    def self_attention(self, x, selected_attn):
        # self attention
        b, c, h, w = self.size
        self_attn = selected_attn.reshape(b, c, h * w)  # b*c*n
        self_attn = self_attn.sum(dim=2).reshape(b, c, 1, 1)  # b*c*1*1

        self_attn_res = self.resConv(x)

        self_attn = self_attn + self_attn_res

        return self_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # sequence attention:a*c*s
        seq_attn = self.sequence_attention(spatial_attn, channel_attn)

        # self-attention
        # self_attn = self.softmax(sk_results)
        # self_attn = x * self_attn  # c,h,w
        self_attn = self.self_attention(x, seq_attn)

        # add attention
        out = seq_attn + self_attn

        # out = self.conv2(out)
        return out

class EX_Module_noselect_par(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module_noselect_par, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        # self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def self_attention(self, x, selected_attn):
        # self attention
        b, c, h, w = self.size
        self_attn = selected_attn.reshape(b, c, h * w)  # b*c*n
        self_attn = self_attn.sum(dim=2).reshape(b, c, 1, 1)  # b*c*1*1

        self_attn_res = self.resConv(x)

        self_attn = self_attn + self_attn_res

        return self_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # parallel attention:a*c+a*s
        par_attn = self.sigmoid(spatial_attn + channel_attn)
        # par_attn = self.sigmoid(x * par_attn)

        # self-attention
        # self_attn = self.softmax(sk_results)
        # self_attn = x * self_attn  # c,h,w
        self_attn = self.self_attention(x, par_attn)

        # add attention
        out = par_attn + self_attn

        # out = self.conv2(out)
        return out