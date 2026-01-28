"""
SSIM Loss for Structural Similarity
====================================
Differentiable SSIM loss for image quality optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def create_gaussian_window(window_size: int, sigma: float,
                           channels: int) -> torch.Tensor:
    """
    Create a Gaussian window for SSIM computation.

    Args:
        window_size: Size of the window
        sigma: Standard deviation of Gaussian
        channels: Number of image channels

    Returns:
        Gaussian window tensor
    """
    # Create 1D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    # Create 2D window
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t())

    # Expand for channels
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    return window


def ssim(pred: torch.Tensor,
         target: torch.Tensor,
         window_size: int = 11,
         sigma: float = 1.5,
         data_range: float = 1.0,
         size_average: bool = True,
         K: Tuple[float, float] = (0.01, 0.03)) -> torch.Tensor:
    """
    Compute SSIM between two images.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian
        data_range: Range of pixel values (1.0 for [0,1], 255 for [0,255])
        size_average: If True, return mean SSIM
        K: Constants for stability (K1, K2)

    Returns:
        SSIM value(s)
    """
    channels = pred.size(1)

    # Create Gaussian window
    window = create_gaussian_window(window_size, sigma, channels)
    window = window.to(pred.device).type(pred.dtype)

    # Constants
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    # Compute means
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_pred_target

    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def ms_ssim(pred: torch.Tensor,
            target: torch.Tensor,
            window_size: int = 11,
            sigma: float = 1.5,
            data_range: float = 1.0,
            weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Multi-Scale SSIM.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian
        data_range: Range of pixel values
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    Returns:
        MS-SSIM value
    """
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                              device=pred.device, dtype=pred.dtype)

    levels = weights.size(0)
    channels = pred.size(1)

    # Create window
    window = create_gaussian_window(window_size, sigma, channels)
    window = window.to(pred.device).type(pred.dtype)

    K = (0.01, 0.03)
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    msssim_vals = []
    mcs_vals = []

    for i in range(levels):
        # Compute means
        mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
        mu_target = F.conv2d(target, window, padding=window_size // 2, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # Compute variances
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_pred_target

        # Luminance and contrast-structure
        luminance = (2 * mu_pred_target + C1) / (mu_pred_sq + mu_target_sq + C1)
        cs = (2 * sigma_pred_target + C2) / (sigma_pred_sq + sigma_target_sq + C2)

        if i == levels - 1:
            msssim_vals.append((luminance * cs).mean())
        else:
            mcs_vals.append(cs.mean())

        # Downsample
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)

    # Combine scales
    msssim = msssim_vals[0]
    for i, mcs in enumerate(mcs_vals):
        msssim = msssim * (mcs ** weights[i])

    return msssim


class SSIMLoss(nn.Module):
    """
    SSIM Loss (1 - SSIM).

    Optimizing this loss maximizes structural similarity.
    """

    def __init__(self,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 data_range: float = 1.0,
                 size_average: bool = True,
                 channel: int = 3):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation for Gaussian
            data_range: Range of pixel values
            size_average: If True, return mean loss
            channel: Number of image channels
        """
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel

        # Pre-compute window
        window = create_gaussian_window(window_size, sigma, channel)
        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            SSIM loss (1 - SSIM)
        """
        ssim_val = ssim(
            pred, target,
            window_size=self.window_size,
            sigma=self.sigma,
            data_range=self.data_range,
            size_average=self.size_average
        )
        return 1 - ssim_val


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM Loss.

    Better captures structural similarity at multiple scales.
    """

    def __init__(self,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 data_range: float = 1.0,
                 weights: Optional[torch.Tensor] = None):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation for Gaussian
            data_range: Range of pixel values
            weights: Weights for each scale
        """
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

        if weights is None:
            weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        self.register_buffer('weights', weights)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute MS-SSIM loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            MS-SSIM loss (1 - MS-SSIM)
        """
        msssim_val = ms_ssim(
            pred, target,
            window_size=self.window_size,
            sigma=self.sigma,
            data_range=self.data_range,
            weights=self.weights
        )
        return 1 - msssim_val


if __name__ == '__main__':
    # Test SSIM
    print("Testing SSIM Loss...")

    ssim_loss = SSIMLoss()

    pred = torch.randn(2, 3, 256, 256).clamp(0, 1)
    target = torch.randn(2, 3, 256, 256).clamp(0, 1)

    loss = ssim_loss(pred, target)
    print(f"SSIM loss (random): {loss.item():.4f}")

    # Test with identical images
    loss_identical = ssim_loss(pred, pred)
    print(f"SSIM loss (identical): {loss_identical.item():.6f}")

    # Test gradient
    pred.requires_grad = True
    loss = ssim_loss(pred, target)
    loss.backward()
    print(f"Gradient computed: {pred.grad is not None}")

    print("Success!")
