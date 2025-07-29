import time

import torch
import torch.nn.functional as F
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models import MeanScaleHyperprior
from compressai.layers.gdn import GDN
from torch import nn
from einops import rearrange 
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
from src.utils.stream_helper import *
import numpy as np


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """
        1x1 convolution.
    """
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


class WMSA(nn.Module):
    """ 
        Self-attention module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2)) #滚动数组元素
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.output_dim, self.output_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.input_dim, self.output_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.output_dim, self.output_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(trans_x)
        x = x + res
        return x
    
class myResidualBlock(nn.Module):
    """
    Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, stride: int = 1):
        super(myResidualBlock, self).__init__()
        self.conv1 = conv(in_ch, out_ch, kernel_size, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv(out_ch, out_ch, kernel_size, stride=stride)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

class MSCNet_Block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1):
        super(MSCNet_Block, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.split_channels = [7.0/16, 5.0/16, 4.0/16] #小中大尺寸通道数目比例
        self.channels_mini_scale = int(self.input_dim * self.split_channels[0]) #小尺寸通道数目
        self.channels_medium_scale = int(self.input_dim * self.split_channels[1]) #中尺寸通道数目
        self.channels_large_scale = int(self.input_dim * self.split_channels[2]) #大尺寸通道数目

        self.conv1 = conv1x1(self.input_dim, self.input_dim)
        self.conv_mini_scale_1 = conv(in_channels=self.channels_mini_scale, out_channels=self.channels_mini_scale, kernel_size=3, stride=1)
        self.conv_medium_scale_1 = conv(in_channels=self.channels_medium_scale, out_channels=self.channels_medium_scale, kernel_size=5, stride=1)
        self.conv_large_scale_1 = conv(in_channels=self.channels_large_scale, out_channels=self.channels_large_scale, kernel_size=7, stride=1)

        self.conv2 = nn.Sequential(
            conv1x1(self.input_dim, self.input_dim),
            nn.LeakyReLU(),
        )

        self.conv_mini_scale_2 = conv(in_channels=self.channels_mini_scale, out_channels=self.channels_mini_scale, kernel_size=3, stride=1)
        self.conv_medium_scale_2 = conv(in_channels=self.channels_medium_scale, out_channels=self.channels_medium_scale, kernel_size=5, stride=1)
        self.conv_large_scale_2 = conv(in_channels=self.channels_large_scale, out_channels=self.channels_large_scale, kernel_size=7, stride=1)

        self.mutiscale_feature_fusion = nn.Sequential(
            conv1x1(self.input_dim, self.output_dim),
            nn.LeakyReLU(),
            ResidualBlock(self.output_dim, self.output_dim)
        )
    
    def forward(self, x):
        x_mini_scale, x_medium_scale, x_large_scale = torch.split(x, (self.channels_mini_scale, self.channels_medium_scale, self.channels_large_scale), dim=1)
        features_mini_scale = self.conv_mini_scale_1(x_mini_scale)
        features_medium_scale = self.conv_medium_scale_1(x_medium_scale)
        features_large_scale = self.conv_large_scale_1(x_large_scale)

        features = torch.cat([features_mini_scale, features_medium_scale, features_large_scale], dim=1)
        features_mini_scale, features_medium_scale, features_large_scale = torch.split(self.conv2(features), (self.channels_mini_scale, self.channels_medium_scale, self.channels_large_scale), dim=1)
        features_mini_scale = self.conv_mini_scale_2(features_mini_scale)
        features_medium_scale = self.conv_medium_scale_2(features_medium_scale)
        features_large_scale = self.conv_large_scale_2(features_large_scale)

        features = torch.cat([features_mini_scale, features_medium_scale, features_large_scale], dim=1)

        features = self.mutiscale_feature_fusion(features)

        return features
    

class feature_Fusion_Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(feature_Fusion_Block, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 全局特征注意力机制(通道注意力机制)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            conv1x1(self.input_dim, self.input_dim//2),
            nn.LeakyReLU(inplace=True),
            conv1x1(self.input_dim//2, self.input_dim)
        )

        self.sigmoid = nn.Sigmoid()

        #局部特征注意力机制(空间注意力机制)
        self.kernel_size=3 
        self.fc = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=self.kernel_size, padding=self.kernel_size//2, bias=False)

        self.fc_fuse = nn.Sequential(
            ResidualBlock(2*self.input_dim, self.output_dim),
            ResidualBlock(self.output_dim, self.output_dim),
        )

    def forward(self, feature_global, feature_local):
        
        # 全局特征-通道注意力机制
        avg_weight_feature_global = self.shared_MLP(self.avg_pool(feature_global))
        max_weight_feature_global = self.shared_MLP(self.max_pool(feature_global))
        weight_feature_global = avg_weight_feature_global + max_weight_feature_global
        out_feature_global = self.sigmoid(weight_feature_global) * feature_global

        # 局部特征-空间注意力机制
        avg_weight_feature_local = torch.mean(feature_local, dim=1, keepdim=True)
        max_weight_feature_local, _ = torch.max(feature_local, dim=1, keepdim=True)

        weight_feature_local = torch.cat([avg_weight_feature_local, max_weight_feature_local], dim=1)
        out_feature_local = self.sigmoid(self.fc(weight_feature_local)) * feature_local

        features = torch.cat([out_feature_global, out_feature_local], dim=1) 
        features = self.fc_fuse(features) + feature_global + feature_local

        return features
    
class hierarchical_Feature_spatial_attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(hierarchical_Feature_spatial_attention, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, feature_yi, feature_pi):

        # 获取形状
        B, C, H, W = feature_yi.shape
        feature_yi_reshaped =  feature_yi.reshape(B, C, -1) #(B, C, H*W)
        feature_pi_reshaped =  feature_pi.reshape(B, C, -1) #(B, C, H*W)

        #获取注意力权重-空间
        weight = F.softmax(torch.bmm(feature_yi_reshaped.permute(0, 2, 1), feature_pi_reshaped), dim=1) #(B, H*W, H*W)

        attention_output = torch.bmm(feature_yi_reshaped, weight)

        spatial_features = attention_output.reshape(feature_yi.size())
        
        return spatial_features


# 层间特征融合模块-通道注意力机制
class hierarchical_Feature_channel_attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(hierarchical_Feature_channel_attention, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, feature_yi, feature_pi):

        B, C, H, W = feature_yi.shape
        feature_yi_reshaped =  feature_yi.reshape(B, C, -1) #(B, C, H*W)
        feature_pi_reshaped =  feature_pi.reshape(B, C, -1) #(B, C, H*W)

        #获取注意力权重-通道
        weight = F.softmax(torch.bmm(feature_yi_reshaped, feature_pi_reshaped.permute(0, 2, 1)), dim=1) #(B, C, C)
        
        attention_output = torch.bmm(weight, feature_yi_reshaped) #(B, C, H*W)

        channel_features = attention_output.reshape(feature_yi.size())
        
        return channel_features


class hierarchical_Feature_Fusion_Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(hierarchical_Feature_Fusion_Block, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 对先前层特征yi进行下采样
        self.ResidualBlockWithStride = ResidualBlockWithStride(self.output_dim, self.output_dim, stride=2)

        # 层间特征融合模块-空间注意力机制
        self.hierarchical_Feature_spatial_attention = hierarchical_Feature_spatial_attention(self.output_dim, self.output_dim)
        # 层间特征融合模块-通道注意力机制
        self.hierarchical_Feature_channel_attention = hierarchical_Feature_channel_attention(self.output_dim, self.output_dim)

        self.hierarchical_Feature_Fuse = nn.Sequential(
            ResidualBlockWithStride(2*self.output_dim, self.output_dim, stride=2),
            AttentionBlock(self.output_dim),
            ResidualBlock(self.output_dim, self.output_dim),
        )


    def forward(self, feature_yi, feature_pi):

        # 提取空间冗余信息
        spatial_feature = self.hierarchical_Feature_spatial_attention(feature_yi, feature_pi)
        #提取通道冗余信息
        channel_feature = self.hierarchical_Feature_channel_attention(feature_yi, feature_pi)

        feature = self.hierarchical_Feature_Fuse(torch.cat([0.25*(spatial_feature + channel_feature)+ 0.5*feature_yi, feature_pi], dim=1))
        
        
        return feature


class myFeatureCombine_version1(nn.Module):
    def __init__(self, N, M) -> None:
        super().__init__()
        self.input_dim, self.output_dim = N, M # 特征的输入和输出维度
        config = [2, 2, 2, 2] # 为了配置窗口注意力和滑动窗口注意力机制的参数
        self.head_dim = [16, 32, 32, 16] # 每一个 swin transformer层维度
        drop_path_rate = 0.0
        begin = 0
        self.window_size = 8 # 窗口大小
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        # 提取p2层全局、局部特征，融合局部和全局特征，并融合当前层和先前层特征
        self.p2Encoder_global = nn.Sequential(
            ConvTransBlock(self.input_dim, self.output_dim, self.head_dim[0], self.window_size, dpr[0+begin], 'W'),
            ConvTransBlock(self.output_dim, self.output_dim, self.head_dim[0], self.window_size, dpr[0+begin], 'SW')
        )
        self.p2Encoder_local = MSCNet_Block(self.input_dim, self.output_dim)
        self.p2Encoder_fuse = feature_Fusion_Block(self.output_dim, self.output_dim)
        self.p2Encoder_fuse_previous = nn.Sequential(
            ResidualBlockWithStride(self.output_dim, self.output_dim, stride=2),
            ResidualBlock(self.output_dim, self.output_dim)
        )

        # 提取p3层全局、局部特征，融合局部和全局特征，并融合当前层和先前层特征
        self.p3Encoder_global = nn.Sequential(
            ConvTransBlock(self.input_dim, self.output_dim, self.head_dim[1], self.window_size, dpr[0+begin], 'W'),
            ConvTransBlock(self.output_dim, self.output_dim, self.head_dim[1], self.window_size, dpr[0+begin], 'SW')
        )
        self.p3Encoder_local = MSCNet_Block(self.input_dim, self.output_dim)
        self.p3Encoder_fuse = feature_Fusion_Block(self.output_dim, self.output_dim)
        self.p3Encoder_fuse_previous = hierarchical_Feature_Fusion_Block(self.output_dim, self.output_dim)

        # 提取p4层全局、局部特征，融合局部和全局特征，并融合当前层和先前层特征
        self.p4Encoder_global = nn.Sequential(
            ConvTransBlock(self.input_dim, self.output_dim, self.head_dim[2], self.window_size, dpr[0+begin], 'W'),
            ConvTransBlock(self.output_dim, self.output_dim, self.head_dim[2], self.window_size, dpr[0+begin], 'SW')
        )
        self.p4Encoder_local = MSCNet_Block(self.input_dim, self.output_dim)
        self.p4Encoder_fuse = feature_Fusion_Block(self.output_dim, self.output_dim)
        self.p4Encoder_fuse_previous = hierarchical_Feature_Fusion_Block(self.output_dim, self.output_dim)

        # 提取p5层全局、局部特征，融合局部和全局特征，并融合当前层和先前层特征
        self.p5Encoder_global = nn.Sequential(
            ConvTransBlock(self.input_dim, self.output_dim, self.head_dim[3], self.window_size, dpr[0+begin], 'W'),
            ConvTransBlock(self.output_dim, self.output_dim, self.head_dim[3], self.window_size, dpr[0+begin], 'SW')
        )
        self.p5Encoder_local = MSCNet_Block(self.input_dim, self.output_dim)
        self.p5Encoder_fuse = feature_Fusion_Block(self.output_dim, self.output_dim)
        self.p5Encoder_fuse_previous = hierarchical_Feature_Fusion_Block(self.output_dim, self.output_dim)

    def forward(self, p_layer_features):
        # p_layer_features contains padded features p2, p3, p4, p5
        # p2：torch.Size([12, 256, 128, 128])
        # p3：torch.Size([12, 256, 64, 64])
        # p4：torch.Size([12, 256, 32, 32]) 
        # p5：torch.Size([12, 256, 16, 16])
        p2, p3, p4, p5 = tuple(p_layer_features) # 获取层级特征(原始数据)

        # 获取p2层特征
        p2_feature_global = self.p2Encoder_global(p2) # p2层全局特征 torch.Size([12, 128, 128, 128])
        p2_feature_local = self.p2Encoder_local(p2) # p2层局部特征 torch.Size([12, 128, 128, 128])
        p2_feature = self.p2Encoder_fuse(p2_feature_global, p2_feature_local) #p2层紧凑特征 torch.Size([12, 128, 128, 128])
        # 融合先前层(-)和p2层特征
        y = self.p2Encoder_fuse_previous(p2_feature) # torch.Size([12, 128, 64, 64])

        # 获取p3层特征
        p3_feature_global = self.p3Encoder_global(p3) #p3层全局特征 torch.Size([12, 128, 64, 64])
        p3_feature_local = self.p3Encoder_local(p3) #p3层局部特征 torch.Size([12, 128, 64, 64])
        p3_feature = self.p3Encoder_fuse(p3_feature_global, p3_feature_local) #p3层紧凑特征 torch.Size([12, 128, 64, 64])
        # 融合先前层(p2)和p3层特征
        y = self.p3Encoder_fuse_previous(y, p3_feature) # torch.Size([12, 128, 32, 32])

        # 获取p4层特征
        p4_feature_global = self.p4Encoder_global(p4) #p4层全局特征 torch.Size([12, 128, 32, 32])
        p4_feature_local = self.p4Encoder_local(p4) #p4层局部特征 torch.Size([12, 128, 32, 32])
        p4_feature = self.p4Encoder_fuse(p4_feature_global, p4_feature_local) #p4层紧凑特征 torch.Size([12, 128, 32, 32])
        # 融合先前层(p2,p3)和p4层特征
        y = self.p4Encoder_fuse_previous(y, p4_feature) #torch.Size([12, 128, 16, 16])

        # 获取p5层特征
        p5_feature_global = self.p5Encoder_global(p5) #p5层全局特征 torch.Size([12, 128, 16, 16])
        p5_feature_local = self.p5Encoder_local(p5) #p5层局部特征 torch.Size([12, 128, 16, 16])
        p5_feature = self.p5Encoder_fuse(p5_feature_global, p5_feature_local) #p5层紧凑特征 torch.Size([12, 128, 16, 16])
        # 融合先前层(p2,p3,p4)和p5层特征
        y = self.p5Encoder_fuse_previous(y, p5_feature) #torch.Size([12, 128, 8, 8])

        return y


class FeatureCombine(nn.Module):
    def __init__(self, N, M) -> None:
        super().__init__()
        self.p2Encoder = nn.Sequential(
            ResidualBlockWithStride(N, M, stride=2),
            ResidualBlock(M, M),
        )

        self.p3Encoder = nn.Sequential(
            ResidualBlockWithStride(N + M, M, stride=2),
            AttentionBlock(M),
            ResidualBlock(M, M),
        )

        self.p4Encoder = nn.Sequential(
            ResidualBlockWithStride(N + M, M, stride=2),
            ResidualBlock(M, M),
        )

        self.p5Encoder = nn.Sequential(
            conv3x3(N + M, M, stride=2),
            AttentionBlock(M),
        )

    def forward(self, p_layer_features):
        # p_layer_features contains padded features p2, p3, p4, p5
        p2, p3, p4, p5 = tuple(p_layer_features)
        y = self.p2Encoder(p2)
        y = self.p3Encoder(torch.cat([y, p3], dim=1))
        y = self.p4Encoder(torch.cat([y, p4], dim=1))
        y = self.p5Encoder(torch.cat([y, p5], dim=1))
        return y


class FeatureSynthesis(nn.Module):
    def __init__(self, N, M) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, N) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(N * 2, N, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )

        self.p4Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )

        self.p3Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            AttentionBlock(M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )
        self.p2Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            AttentionBlock(M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            subpel_conv3x3(M, N, 2),
        )

        self.decoder_attention = AttentionBlock(M)

        self.fmb23 = FeatureMixingBlock(N)
        self.fmb34 = FeatureMixingBlock(N)
        self.fmb45 = FeatureMixingBlock(N)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p2 = self.p2Decoder(y_hat)
        p3 = self.fmb23(p2, self.p3Decoder(y_hat))
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p2, p3, p4, p5]


class FeatureCompressor(MeanScaleHyperprior):
    def __init__(self, N=256, M=128, **kwargs):
        super().__init__(M, M, **kwargs)

        # self.g_a = FeatureCombine(N, M) # L-MSFC
        self.g_a = myFeatureCombine_version1(N, M)
        self.g_s = FeatureSynthesis(N, M)

        self.h_a = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M, M, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M * 3 // 2, M * 2),
        )

        self.p6Decoder = nn.Sequential(nn.MaxPool2d(1, stride=2))


    def forward(self, features, decode = False, output_path=None):

        _, _, p2_h, p2_w = features[:-1][0].shape
        # features: [p2, p3, p4, p5, p6]
        if decode:
            self.update()
            encode_info = self.encode_decode(features, output_path, p2_h, p2_w)
            return encode_info
        else:
            features = features[:-1]
            _, _, p2_h, p2_w = features[0].shape
            pad_info = self.cal_feature_padding_size((p2_h, p2_w))
            features = self.feature_padding(features, pad_info)

            y = self.g_a(features)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)

            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

            recon_p_layer_features = self.g_s(y_hat)
            recon_p_layer_features = self.feature_unpadding(
                recon_p_layer_features, pad_info
            )

            p6 = self.p6Decoder(
                recon_p_layer_features[3]
            )  # p6 is generated from p5 directly

            recon_p_layer_features.append(p6)

            return {
                "features": recon_p_layer_features,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

    def compress(self, features):  # features: [p2, p3, p4, p5, p6]
        features = features[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)
        y = self.g_a(features)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, p2_h, p2_w):
        assert isinstance(strings, list) and len(strings) == 2
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        padded_p2_h = pad_info["padded_size"][0][0]
        padded_p2_w = pad_info["padded_size"][0][1]
        z_shape = get_downsampled_shape(padded_p2_h, padded_p2_w, 64)
        z_hat = self.entropy_bottleneck.decompress(strings[1], z_shape)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        recon_p_layer_features = self.g_s(y_hat)
        recon_p_layer_features = self.feature_unpadding(
            recon_p_layer_features, pad_info
        )
        p6 = self.p6Decoder(
            recon_p_layer_features[3]
        )  # p6 is generated from p5 directly
        recon_p_layer_features.append(p6)
        return {"features": recon_p_layer_features}

    def encode_decode(self, features, output_path, p2_height, p2_width):
        encoding_time_start = time.time()
        encoded = self.encode(features, output_path, p2_height, p2_width)
        encoding_time = time.time() - encoding_time_start
        decoding_time_start = time.time()
        decoded = self.decode(output_path)
        decoding_time = time.time() - decoding_time_start
        encoded.update(decoded)
        encoded["encoding_time"] = encoding_time
        encoded["decoding_time"] = decoding_time
        return encoded

    def encode(self, features, output_path, p2_height, p2_width):
        encoded = self.compress(features)
        y_string = encoded["strings"][0][0]
        z_string = encoded["strings"][1][0]

        encode_feature(p2_height, p2_width, y_string, z_string, output_path)
        bits = filesize(output_path) * 8
        summary = {
            "bit": bits,
            "bit_y": len(y_string) * 8,
            "bit_z": len(z_string) * 8,
        }
        encoded.update(summary)
        return encoded

    def decode(self, input_path):
        p2_height, p2_width, y_string, z_string = decode_feature(input_path)
        decoded = self.decompress([y_string, z_string], p2_height, p2_width)
        return decoded

    def cal_feature_padding_size(self, p2_shape):
        ps_list = [64, 32, 16, 8]
        ori_size = []
        paddings = []
        unpaddings = []
        padded_size = []

        ori_size.append(p2_shape)
        for i in range(len(ps_list) - 1):
            h, w = ori_size[-1]
            ori_size.append(((h + 1) // 2, (w + 1) // 2))

        for i, ps in enumerate(ps_list):
            h = ori_size[i][0]
            w = ori_size[i][1]

            h_pad_len = ps - h % ps if h % ps != 0 else 0
            w_pad_len = ps - w % ps if w % ps != 0 else 0

            paddings.append(
                (
                    w_pad_len // 2,
                    w_pad_len - w_pad_len // 2,
                    h_pad_len // 2,
                    h_pad_len - h_pad_len // 2,
                )
            )
            unpaddings.append(
                (
                    0 - (w_pad_len // 2),
                    0 - (w_pad_len - w_pad_len // 2),
                    0 - (h_pad_len // 2),
                    0 - (h_pad_len - h_pad_len // 2),
                )
            )

        for i, p in enumerate(paddings):
            h = ori_size[i][0]
            w = ori_size[i][1]
            h_pad_len = p[2] + p[3]
            w_pad_len = p[0] + p[1]
            padded_size.append((h + h_pad_len, w + w_pad_len))

        return {
            "ori_size": ori_size,
            "paddings": paddings,
            "unpaddings": unpaddings,
            "padded_size": padded_size,
        }

    def feature_padding(self, features, pad_info):
        p2, p3, p4, p5 = features
        paddings = pad_info["paddings"]

        p2 = F.pad(p2, paddings[0], mode="reflect")
        p3 = F.pad(p3, paddings[1], mode="reflect")
        p4 = F.pad(p4, paddings[2], mode="reflect")
        p5 = F.pad(p5, paddings[3], mode="reflect")
        return [p2, p3, p4, p5]

    def feature_unpadding(self, features, pad_info):
        p2, p3, p4, p5 = features
        unpaddings = pad_info["unpaddings"]

        p2 = F.pad(p2, unpaddings[0])
        p3 = F.pad(p3, unpaddings[1])
        p4 = F.pad(p4, unpaddings[2])
        p5 = F.pad(p5, unpaddings[3])
        return [p2, p3, p4, p5]
