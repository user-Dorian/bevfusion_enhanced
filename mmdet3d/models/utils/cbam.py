# Copyright (c) OpenMMLab. All rights reserved.
"""CBAM: Convolutional Block Attention Module.

Reference:
    CBAM: Convolutional Block Attention Module
    https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html
"""

import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


class ChannelAttention(nn.Module):
    """Channel attention module.

    Args:
        channels (int): The number of input channels.
        ratio (int): The reduction ratio for the bottleneck layer.
            Default: 16.
    """

    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module.

    Args:
        kernel_size (int): The kernel size of the convolution.
            Default: 7.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 5, 7, 9), f'kernel_size must be 3, 5, 7, or 9, but got {kernel_size}'
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function."""
        # Channel-wise max and average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        # Convolution
        x = self.conv(x)
        return self.sigmoid(x) * x


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module.
    
    This module applies both channel and spatial attention sequentially.
    
    Args:
        channels (int): The number of input channels.
        ratio (int): The reduction ratio for channel attention.
            Default: 16.
        kernel_size (int): The kernel size for spatial attention.
            Default: 7.
    """

    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input features with shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Attended features with shape (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
