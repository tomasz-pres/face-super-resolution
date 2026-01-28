"""
ESRGAN Baseline Model
======================
Pre-trained ESRGAN/Real-ESRGAN wrapper for baseline comparison.
"""

import os
from pathlib import Path
from typing import Optional, Union, Tuple
import urllib.request

import torch
import torch.nn as nn
import numpy as np


class RRDBNet(nn.Module):
    """
    RRDB Network for ESRGAN.

    Residual in Residual Dense Block architecture.
    """

    def __init__(self,
                 num_in_ch: int = 3,
                 num_out_ch: int = 3,
                 num_feat: int = 64,
                 num_block: int = 23,
                 num_grow_ch: int = 32,
                 scale: int = 4):
        super().__init__()

        self.scale = scale

        # First convolution
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # RRDB blocks
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )

        # After body convolution
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample
        feat = self.lrelu(self.conv_up1(
            nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(
            nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class ESRGANBaseline(nn.Module):
    """
    ESRGAN Baseline Model Wrapper.

    Provides consistent interface for inference with pre-trained weights.
    """

    # Pre-trained weight URLs
    WEIGHT_URLS = {
        'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'ESRGAN_x4': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
    }

    def __init__(self,
                 model_name: str = 'RealESRGAN_x4plus',
                 scale: int = 4,
                 device: Optional[str] = None,
                 weights_dir: str = 'checkpoints/pretrained'):
        """
        Args:
            model_name: Name of pre-trained model
            scale: Upscaling factor
            device: Device to use ('cuda', 'cpu', or None for auto)
            weights_dir: Directory to store downloaded weights
        """
        super().__init__()

        self.scale = scale
        self.model_name = model_name
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create model
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )

        # Load weights
        self._load_weights()

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_weights(self) -> None:
        """Download and load pre-trained weights."""
        weights_path = self.weights_dir / f'{self.model_name}.pth'

        if not weights_path.exists():
            if self.model_name in self.WEIGHT_URLS:
                print(f"Downloading {self.model_name} weights...")
                url = self.WEIGHT_URLS[self.model_name]
                urllib.request.urlretrieve(url, weights_path)
                print(f"Saved to {weights_path}")
            else:
                print(f"Warning: No pre-trained weights found for {self.model_name}")
                print("Using randomly initialized weights (for testing only)")
                return

        # Load weights
        state_dict = torch.load(weights_path, map_location='cpu')

        # Handle different state dict formats
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']

        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W) in range [0, 1]

        Returns:
            Super-resolved tensor (B, 3, H*scale, W*scale)
        """
        return self.model(x)

    @torch.no_grad()
    def inference(self,
                  lr_image: Union[np.ndarray, torch.Tensor],
                  return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Run inference on a single image.

        Args:
            lr_image: Low-resolution image (H, W, 3) uint8 or (3, H, W) tensor
            return_numpy: If True, return numpy array

        Returns:
            Super-resolved image
        """
        # Preprocess
        if isinstance(lr_image, np.ndarray):
            # Convert to tensor
            if lr_image.dtype == np.uint8:
                lr_image = lr_image.astype(np.float32) / 255.0
            # HWC to CHW
            lr_tensor = torch.from_numpy(lr_image.transpose(2, 0, 1)).unsqueeze(0)
        else:
            lr_tensor = lr_image
            if lr_tensor.dim() == 3:
                lr_tensor = lr_tensor.unsqueeze(0)

        # Move to device
        lr_tensor = lr_tensor.to(self.device)

        # Forward pass
        sr_tensor = self.forward(lr_tensor)

        # Postprocess
        sr_tensor = torch.clamp(sr_tensor, 0, 1)

        if return_numpy:
            sr_image = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr_image = (sr_image * 255).astype(np.uint8)
            return sr_image
        else:
            return sr_tensor.squeeze(0)

    @torch.no_grad()
    def inference_batch(self, lr_batch: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of images.

        Args:
            lr_batch: Batch of LR images (B, 3, H, W) in range [0, 1]

        Returns:
            Batch of SR images (B, 3, H*scale, W*scale)
        """
        lr_batch = lr_batch.to(self.device)
        sr_batch = self.forward(lr_batch)
        return torch.clamp(sr_batch, 0, 1)

    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'name': self.model_name,
            'scale': self.scale,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
            'device': str(self.device)
        }


def create_esrgan_baseline(device: Optional[str] = None,
                           weights_dir: str = 'checkpoints/pretrained') -> ESRGANBaseline:
    """
    Factory function to create ESRGAN baseline model.

    Args:
        device: Device to use
        weights_dir: Directory for weights

    Returns:
        ESRGANBaseline model
    """
    return ESRGANBaseline(
        model_name='RealESRGAN_x4plus',
        scale=4,
        device=device,
        weights_dir=weights_dir
    )


if __name__ == '__main__':
    # Test the model
    print("Creating ESRGAN baseline model...")
    model = create_esrgan_baseline()

    info = model.get_model_info()
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test inference
    print("\nTesting inference...")
    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(dummy_input.to(model.device))
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Success!")
