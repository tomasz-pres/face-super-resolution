"""
Neural Network Building Blocks
==============================
Custom blocks for FaceEnhanceNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def icnr_init(tensor: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """
    ICNR (Initialization Checkered Noise Reduction) for PixelShuffle.

    Fills the tensor with values that avoid checkerboard artifacts
    when using PixelShuffle upsampling.

    Args:
        tensor: Weight tensor of shape (out_channels, in_channels, kH, kW)
        scale_factor: Upscaling factor

    Returns:
        Initialized tensor
    """
    out_channels, in_channels, kH, kW = tensor.shape
    sub_channels = out_channels // (scale_factor ** 2)

    # Initialize sub-kernel with Kaiming
    sub_kernel = torch.empty(sub_channels, in_channels, kH, kW)
    nn.init.kaiming_normal_(sub_kernel, mode='fan_out', nonlinearity='relu')

    # Repeat for each sub-pixel
    kernel = sub_kernel.repeat_interleave(scale_factor ** 2, dim=0)

    with torch.no_grad():
        tensor.copy_(kernel)

    return tensor


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style Channel Attention.

    Learns to recalibrate channel-wise feature responses by
    explicitly modeling interdependencies between channels.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        """
        Args:
            num_channels: Number of input channels
            reduction_ratio: Reduction ratio for bottleneck
        """
        super().__init__()

        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = max(num_channels // reduction_ratio, 8)

        # Squeeze: Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Recalibrated tensor (B, C, H, W)
        """
        b, c, _, _ = x.size()

        # Squeeze
        y = self.global_pool(x).view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block.

    Combines residual learning with channel attention for
    effective feature extraction in super-resolution.

    Uses residual scaling (default 0.1) as in RCAN paper for training stability.
    """

    def __init__(self,
                 num_channels: int = 64,
                 kernel_size: int = 3,
                 reduction_ratio: int = 4,
                 bias: bool = True,
                 res_scale: float = 0.2):
        """
        Args:
            num_channels: Number of feature channels
            kernel_size: Convolution kernel size
            reduction_ratio: Channel attention reduction ratio
            bias: Whether to use bias in convolutions
            res_scale: Residual scaling factor for stability (ESRGAN uses 0.2)
        """
        super().__init__()

        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size,
            padding=kernel_size // 2, bias=bias
        )
        self.prelu = nn.PReLU(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size,
            padding=kernel_size // 2, bias=bias
        )

        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        residual = x

        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)

        # Channel attention
        out = self.channel_attention(out)

        # Residual connection with scaling for stability
        return out * self.res_scale + residual


class ResidualGroup(nn.Module):
    """
    Group of RCAB blocks with residual connection.
    """

    def __init__(self,
                 num_channels: int = 64,
                 num_blocks: int = 4,
                 kernel_size: int = 3,
                 reduction_ratio: int = 4,
                 res_scale: float = 0.2):
        """
        Args:
            num_channels: Number of feature channels
            num_blocks: Number of RCAB blocks in the group
            kernel_size: Convolution kernel size
            reduction_ratio: Channel attention reduction ratio
            res_scale: Residual scaling factor for stability
        """
        super().__init__()

        blocks = [
            RCAB(num_channels, kernel_size, reduction_ratio, res_scale=res_scale)
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.blocks(x)
        out = self.conv(out)
        return out + residual


class PixelShuffleUpsample(nn.Module):
    """
    Pixel Shuffle based upsampling block.

    Efficient sub-pixel convolution for upscaling.
    Uses ICNR initialization to prevent checkerboard artifacts.
    """

    def __init__(self,
                 in_channels: int,
                 scale_factor: int = 2):
        """
        Args:
            in_channels: Number of input channels
            scale_factor: Upscaling factor (2 or 4)
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(
            in_channels, in_channels * (scale_factor ** 2),
            kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU(in_channels)

        # Apply ICNR initialization
        icnr_init(self.conv.weight, scale_factor)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


class UpsampleModule(nn.Module):
    """
    Multi-stage upsampling module using PixelShuffle.

    For 4x upscaling, uses two 2x stages.
    """

    def __init__(self,
                 num_channels: int = 64,
                 scale_factor: int = 4):
        """
        Args:
            num_channels: Number of feature channels
            scale_factor: Total upscaling factor (must be power of 2)
        """
        super().__init__()

        self.scale_factor = scale_factor

        # Calculate number of 2x stages needed
        num_stages = 0
        temp_scale = scale_factor
        while temp_scale > 1:
            temp_scale //= 2
            num_stages += 1

        # Create upsampling stages
        stages = []
        for _ in range(num_stages):
            stages.append(PixelShuffleUpsample(num_channels, scale_factor=2))
        self.stages = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)


def initialize_weights(module: nn.Module, scale: float = 0.1) -> None:
    """
    Initialize module weights.

    Uses Kaiming initialization for Conv2d and PReLU layers.

    Args:
        module: Module to initialize
        scale: Scale factor for initialization
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            m.weight.data *= scale
