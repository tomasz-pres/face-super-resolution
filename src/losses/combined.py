"""
Combined Loss Functions for Super-Resolution
==============================================
Flexible loss combination with component tracking.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .perceptual import PerceptualLoss
from .ssim_loss import SSIMLoss, MSSSIMLoss


@dataclass
class LossConfig:
    """Configuration for combined loss."""

    # Loss weights
    l1_weight: float = 1.0
    l2_weight: float = 0.0
    perceptual_weight: float = 0.01
    ssim_weight: float = 0.1
    ms_ssim_weight: float = 0.0

    # Charbonnier loss settings (replaces L1 when enabled)
    use_charbonnier: bool = False
    charbonnier_eps: float = 1e-3

    # Perceptual loss settings
    perceptual_layers: list = field(default_factory=lambda: ['conv3_4', 'conv4_4'])

    # SSIM settings
    ssim_window_size: int = 11


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) Loss."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """L2 (Mean Squared Error) Loss."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1).

    More robust to outliers than L1.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for super-resolution training.

    Combines:
    - L1 loss (pixel-wise reconstruction)
    - L2 loss (optional, smoother gradients)
    - Perceptual loss (VGG feature matching)
    - SSIM loss (structural similarity)
    - MS-SSIM loss (multi-scale structural similarity)

    Tracks individual loss components for analysis.
    """

    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        """
        Args:
            config: Loss configuration
            **kwargs: Override config parameters
        """
        super().__init__()

        if config is None:
            config = LossConfig()

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Initialize losses
        self.losses = nn.ModuleDict()
        self.weights = {}

        if config.l1_weight > 0:
            # Use Charbonnier loss instead of L1 if enabled
            if config.use_charbonnier:
                self.losses['l1'] = CharbonnierLoss(epsilon=config.charbonnier_eps)
            else:
                self.losses['l1'] = L1Loss()
            self.weights['l1'] = config.l1_weight

        if config.l2_weight > 0:
            self.losses['l2'] = L2Loss()
            self.weights['l2'] = config.l2_weight

        if config.perceptual_weight > 0:
            self.losses['perceptual'] = PerceptualLoss(
                layers=config.perceptual_layers
            )
            self.weights['perceptual'] = config.perceptual_weight

        if config.ssim_weight > 0:
            self.losses['ssim'] = SSIMLoss(
                window_size=config.ssim_window_size
            )
            self.weights['ssim'] = config.ssim_weight

        if config.ms_ssim_weight > 0:
            self.losses['ms_ssim'] = MSSSIMLoss()
            self.weights['ms_ssim'] = config.ms_ssim_weight

        # Track loss history
        self.loss_history: Dict[str, list] = {name: [] for name in self.losses.keys()}
        self.loss_history['total'] = []

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor,
                return_components: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]
            return_components: Whether to return individual loss components

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weighted_loss = self.weights[name] * loss_value

            total_loss = total_loss + weighted_loss
            loss_dict[name] = loss_value.detach()

        loss_dict['total'] = total_loss.detach()

        if return_components:
            return total_loss, loss_dict
        else:
            return total_loss, {}

    def update_weight(self, name: str, weight: float) -> None:
        """Update weight for a specific loss component."""
        if name in self.weights:
            self.weights[name] = weight
        else:
            raise ValueError(f"Unknown loss component: {name}")

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()

    def record_loss(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Record loss values for tracking."""
        for name, value in loss_dict.items():
            if name in self.loss_history:
                self.loss_history[name].append(value.item())

    def get_loss_history(self) -> Dict[str, list]:
        """Get recorded loss history."""
        return self.loss_history.copy()

    def reset_history(self) -> None:
        """Reset loss history."""
        for name in self.loss_history:
            self.loss_history[name] = []


class LossTracker:
    """
    Utility class for tracking and analyzing loss values.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.history: Dict[str, list] = {}
        self.epoch_history: Dict[str, list] = {}

    def update(self, loss_dict: Dict[str, Any]) -> None:
        """Update with new loss values."""
        for name, value in loss_dict.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(
                value.item() if torch.is_tensor(value) else value
            )

    def get_moving_average(self, name: str) -> float:
        """Get moving average for a loss component."""
        if name not in self.history or len(self.history[name]) == 0:
            return 0.0

        values = self.history[name][-self.window_size:]
        return sum(values) / len(values)

    def get_epoch_average(self, name: str) -> float:
        """Get average loss for current epoch."""
        if name not in self.history or len(self.history[name]) == 0:
            return 0.0
        return sum(self.history[name]) / len(self.history[name])

    def end_epoch(self) -> Dict[str, float]:
        """End epoch and record averages."""
        epoch_avgs = {}
        for name, values in self.history.items():
            if values:
                avg = sum(values) / len(values)
                epoch_avgs[name] = avg

                if name not in self.epoch_history:
                    self.epoch_history[name] = []
                self.epoch_history[name].append(avg)

        # Clear iteration history
        self.history = {name: [] for name in self.history}

        return epoch_avgs

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}
        for name, values in self.epoch_history.items():
            if values:
                summary[name] = {
                    'current': values[-1],
                    'best': min(values),
                    'worst': max(values),
                    'mean': sum(values) / len(values),
                }
        return summary

    def to_dict(self) -> Dict[str, list]:
        """Export epoch history to dictionary."""
        return self.epoch_history.copy()


def create_loss_function(
    l1_weight: float = 1.0,
    perceptual_weight: float = 0.01,
    ssim_weight: float = 0.1,
    **kwargs
) -> CombinedLoss:
    """
    Factory function to create combined loss.

    Args:
        l1_weight: Weight for L1 loss
        perceptual_weight: Weight for perceptual loss
        ssim_weight: Weight for SSIM loss
        **kwargs: Additional config parameters

    Returns:
        CombinedLoss instance
    """
    config = LossConfig(
        l1_weight=l1_weight,
        perceptual_weight=perceptual_weight,
        ssim_weight=ssim_weight,
        **kwargs
    )
    return CombinedLoss(config)


if __name__ == '__main__':
    # Test combined loss
    print("Testing Combined Loss...")

    loss_fn = create_loss_function(
        l1_weight=1.0,
        perceptual_weight=0.01,
        ssim_weight=0.1
    )

    pred = torch.randn(2, 3, 256, 256).clamp(0, 1)
    target = torch.randn(2, 3, 256, 256).clamp(0, 1)

    total_loss, components = loss_fn(pred, target)

    print(f"\nLoss components:")
    for name, value in components.items():
        print(f"  {name}: {value.item():.6f}")

    print(f"\nWeights: {loss_fn.get_weights()}")

    # Test gradient flow
    pred.requires_grad = True
    total_loss, _ = loss_fn(pred, target)
    total_loss.backward()
    print(f"\nGradient computed: {pred.grad is not None}")

    # Test loss tracker
    print("\nTesting Loss Tracker...")
    tracker = LossTracker()

    for i in range(10):
        tracker.update(components)

    print(f"Moving average (total): {tracker.get_moving_average('total'):.6f}")

    epoch_avgs = tracker.end_epoch()
    print(f"Epoch averages: {epoch_avgs}")

    print("\nSuccess!")
