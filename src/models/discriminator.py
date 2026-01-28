"""
VGG-Style Discriminator for GAN-based Super-Resolution
=======================================================
Discriminator network for adversarial training of face SR models.
"""

import torch
import torch.nn as nn
from typing import Optional


class VGGStyleDiscriminator(nn.Module):
    """
    VGG-style Discriminator for 256x256 images.

    Architecture follows SRGAN/ESRGAN discriminator design:
    - Progressively increasing channels: 64 -> 128 -> 256 -> 512
    - Strided convolutions for downsampling
    - BatchNorm + LeakyReLU activations
    - Final dense layers for real/fake classification

    Input: 256x256x3 image
    Output: Scalar (real/fake probability or score)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        input_size: int = 256,
        use_bn: bool = True,
        use_sigmoid: bool = False  # False for LSGAN/WGAN, True for vanilla GAN
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (doubles each block)
            input_size: Input image size (256 for our case)
            use_bn: Whether to use BatchNorm
            use_sigmoid: Whether to apply sigmoid at output
        """
        super().__init__()

        self.use_sigmoid = use_sigmoid

        def conv_block(in_ch, out_ch, kernel_size=3, stride=1, use_bn=True):
            """Create a conv block with optional BatchNorm and LeakyReLU."""
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2, bias=not use_bn)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Feature extraction blocks
        # Input: 256x256x3
        self.features = nn.Sequential(
            # Block 1: 256x256 -> 256x256 -> 128x128
            conv_block(in_channels, base_channels, use_bn=False),  # No BN on first layer
            conv_block(base_channels, base_channels, stride=2, use_bn=use_bn),

            # Block 2: 128x128 -> 128x128 -> 64x64
            conv_block(base_channels, base_channels * 2, use_bn=use_bn),
            conv_block(base_channels * 2, base_channels * 2, stride=2, use_bn=use_bn),

            # Block 3: 64x64 -> 64x64 -> 32x32
            conv_block(base_channels * 2, base_channels * 4, use_bn=use_bn),
            conv_block(base_channels * 4, base_channels * 4, stride=2, use_bn=use_bn),

            # Block 4: 32x32 -> 32x32 -> 16x16
            conv_block(base_channels * 4, base_channels * 8, use_bn=use_bn),
            conv_block(base_channels * 8, base_channels * 8, stride=2, use_bn=use_bn),

            # Block 5: 16x16 -> 16x16 -> 8x8
            conv_block(base_channels * 8, base_channels * 8, use_bn=use_bn),
            conv_block(base_channels * 8, base_channels * 8, stride=2, use_bn=use_bn),
        )

        # Calculate feature map size after conv blocks
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        feature_size = input_size // 32  # 256 // 32 = 8

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8 * feature_size * feature_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (B, 3, 256, 256)

        Returns:
            Discriminator score (B, 1)
        """
        features = self.features(x)
        out = self.classifier(features)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out

    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'name': 'VGGStyleDiscriminator',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 ** 2),
        }


class GANLoss(nn.Module):
    """
    GAN Loss wrapper supporting multiple GAN types.

    Supports:
    - 'vanilla': Standard GAN with BCE loss
    - 'lsgan': Least Squares GAN (MSE loss)
    - 'wgan': Wasserstein GAN (no activation)
    """

    def __init__(self, gan_type: str = 'vanilla', real_label: float = 1.0, fake_label: float = 0.0):
        """
        Args:
            gan_type: Type of GAN loss ('vanilla', 'lsgan', 'wgan')
            real_label: Label for real images
            fake_label: Label for fake images
        """
        super().__init__()

        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label

        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgan':
            self.loss = None  # WGAN uses raw scores
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def get_target_tensor(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Create target tensor with same shape as prediction."""
        target_val = self.real_label if is_real else self.fake_label
        return torch.full_like(prediction, target_val)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Calculate GAN loss.

        Args:
            prediction: Discriminator output
            is_real: Whether this is for real or fake images

        Returns:
            GAN loss value
        """
        if self.gan_type == 'wgan':
            # WGAN: maximize for real, minimize for fake
            return -prediction.mean() if is_real else prediction.mean()
        else:
            target = self.get_target_tensor(prediction, is_real)
            return self.loss(prediction, target)


def create_discriminator(
    input_size: int = 256,
    base_channels: int = 64,
    use_bn: bool = True,
    **kwargs
) -> VGGStyleDiscriminator:
    """
    Factory function to create discriminator.

    Args:
        input_size: Input image size
        base_channels: Base number of channels
        use_bn: Whether to use BatchNorm

    Returns:
        VGGStyleDiscriminator instance
    """
    return VGGStyleDiscriminator(
        in_channels=3,
        base_channels=base_channels,
        input_size=input_size,
        use_bn=use_bn,
        use_sigmoid=False  # We use BCEWithLogitsLoss
    )


if __name__ == '__main__':
    # Test discriminator
    print("Testing VGGStyleDiscriminator...")

    disc = create_discriminator(input_size=256)
    info = disc.get_model_info()

    print(f"\nDiscriminator Info:")
    for k, v in info.items():
        if isinstance(v, int) and v > 1000:
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v}")

    # Test forward pass
    dummy_input = torch.randn(4, 3, 256, 256)
    output = disc(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test GAN loss
    print("\nTesting GANLoss...")
    gan_loss = GANLoss(gan_type='vanilla')

    real_loss = gan_loss(output, is_real=True)
    fake_loss = gan_loss(output, is_real=False)
    print(f"Real loss: {real_loss.item():.4f}")
    print(f"Fake loss: {fake_loss.item():.4f}")

    print("\nSuccess!")
