import math
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np
import pickle
import cv2
import os

def exists(x):
    return x is not None

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)



class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)

        # 根据需要使用不同的注意力机制
        if with_attn == "SpatialSelfAttention":
            self.with_attn = SpatialSelfAttention(dim_out)
        elif with_attn == "SelfAttention":
            self.with_attn = SelfAttention(dim_out)
        elif with_attn == "MixedAttention":
            self.with_attn = MixedAttention(dim_out)
        elif with_attn == "MaskGuidedAttention":
            self.with_attn = MaskGuidedAttention(dim_out)
        else:
            self.with_attn = False

    def forward(self, x, mask,time_emb):
        # 通过 Resnet block
        x = self.res_block(x, time_emb)

        # 如果使用注意力机制，传递 mask
        if self.with_attn:
            x = self.with_attn(x,mask)  # 确保传递 mask
        return x



class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = nn.LayerNorm(in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.dropout = nn.Dropout(0.1)  # 添加 Dropout 正则化

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)  # 应用 Dropout 正则化
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


# 改进点2
# 空间自注意力 残差连接
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class MixedAttention(nn.Module):
    def __init__(self, in_channels, n_head=4, norm_groups=32):
        super(MixedAttention, self).__init__()
        self.self_attention = SelfAttention(in_channels, n_head, norm_groups)
        self.spatial_attention = SpatialSelfAttention(in_channels)
        # 初始化权重为可学习的参数
        self.self_attn_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        x_self_attention = self.self_attention(x)
        x_spatial_attention = self.spatial_attention(x)
        # 使用可学习的权重
        return self.self_attn_weight * x_self_attention + (1 - self.self_attn_weight) * x_spatial_attention


class MaskGuidedAttention(nn.Module):
    def __init__(self, in_channels, n_head=4, norm_groups=32):
        super(MaskGuidedAttention, self).__init__()
        self.self_attention = SelfAttention(in_channels, n_head, norm_groups)
        self.spatial_attention = SpatialSelfAttention(in_channels)

        # 自适应通道和空间注意力权重
        self.channel_attn_weight = nn.Parameter(torch.tensor(0.5))
        self.spatial_attn_weight = nn.Parameter(torch.tensor(0.5))

        # 动态掩膜权重
        self.mask_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，让网络在阴影和非阴影区域之间平衡

        # 多尺度卷积
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        # 通道融合卷积
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 通道注意力机制
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        # 计算自注意力和空间注意力
        x_self_attention = self.self_attention(x)
        x_spatial_attention = self.spatial_attention(x)

        # 自适应调整通道和空间注意力
        attention_fused = self.channel_attn_weight * x_self_attention + self.spatial_attn_weight * x_spatial_attention

        # 多尺度卷积
        x_3x3 = self.conv3x3(x)
        x_5x5 = self.conv5x5(x)

        # 将多尺度卷积结果与注意力结果结合
        attention_fused = attention_fused + x_3x3 + x_5x5

        # 调整掩膜尺寸与 attention_fused 一致
        mask_resized = F.interpolate(mask, size=attention_fused.shape[2:], mode='bilinear', align_corners=False)

        # 利用掩膜引导注意力
        mask_resized = mask_resized.expand_as(attention_fused)
        shadow_aware_attention = self.mask_weight * attention_fused * mask_resized + (
                1 - self.mask_weight) * attention_fused * (1 - mask_resized)

        # 通道注意力机制
        channel_weight = self.global_avg_pool(shadow_aware_attention)
        channel_weight = self.channel_attention(channel_weight)

        # 对每个通道应用权重
        shadow_aware_attention = shadow_aware_attention * channel_weight

        # 最终输出
        x_out = self.conv1x1(attention_fused)
        return x_out + x  # 残差连接


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn="MaskGuidedAttention"),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.mask_tail = MEM()

    def forward(self, x, time):
        # x_lr = x[:, :3, :, :]
        x_mask = x[:, 3, :, :].unsqueeze(1)
        # x_noisy = x[:, 4:, :, :]
        # updated_mask = self.mask_update(x_noisy, x_mask)
        # # x_updated_mask = updated_mask.detach()
        # x = torch.cat((x_lr, updated_mask, x_noisy), dim=1)

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x,x_mask, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x,x_mask, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                feat = feats.pop()
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                x = layer(torch.cat((x, feat), dim=1), x_mask,t)

            else:
                x = layer(x)

        return self.final_conv(x), self.mask_tail(x)


class MEM(nn.Module):
    def __init__(self):
        super(MEM, self).__init__()
        # 原始卷积和空洞卷积
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dilated_conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)  # 空洞卷积
        self.dilated_conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)  # 空洞卷积

        # 金字塔池化模块
        self.pyramid_pool1 = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.pyramid_pool2 = nn.AdaptiveAvgPool2d(2)  # 2x2 平均池化
        self.pyramid_pool3 = nn.AdaptiveAvgPool2d(4)  # 4x4 平均池化

        # 通道压缩卷积层，将通道数从 256 压缩到 67
        self.channel_compress = nn.Conv2d(256, 67, 1, 1)

        # 最终卷积层
        self.final_conv = nn.Conv2d(67, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        # 调整 identity 通道数的卷积层
        self.match_channels = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        identity = x  # 保存原始输入作为恒等（残差）连接

        # 原始卷积和空洞卷积
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.dilated_conv1(x1))
        x3 = self.relu(self.dilated_conv2(x2))

        # 多尺度特征融合
        x_fused = x1 + x2 + x3

        # 金字塔池化
        p1 = self.pyramid_pool1(x_fused)  # 全局池化
        p2 = self.pyramid_pool2(x_fused)  # 2x2池化
        p3 = self.pyramid_pool3(x_fused)  # 4x4池化

        # 上采样到与 x_fused 一样的尺寸
        p1_up = F.interpolate(p1, size=x_fused.shape[2:], mode='bilinear', align_corners=True)
        p2_up = F.interpolate(p2, size=x_fused.shape[2:], mode='bilinear', align_corners=True)
        p3_up = F.interpolate(p3, size=x_fused.shape[2:], mode='bilinear', align_corners=True)

        # 将池化结果与原始特征图拼接
        x_fused = torch.cat([x_fused, p1_up, p2_up, p3_up], dim=1)

        # 使用通道压缩层将通道数从 256 压缩到 67
        x_fused = self.channel_compress(x_fused)

        # 最终卷积生成 refined mask
        out = self.sigmoid(self.final_conv(x_fused))

        # 将 identity 调整为与 out 相同的通道数
        identity = self.match_channels(identity)

        # 添加残差连接
        out = out + identity

        return out

