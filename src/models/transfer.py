"""
Transfer Learning Model for Face Super-Resolution
==================================================
Fine-tuned ESRGAN backbone with face-specific head.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from .esrgan import RRDBNet, RRDB, ResidualDenseBlock
from .blocks import RCAB, UpsampleModule


class TrainingStage(Enum):
    """Training stages for progressive unfreezing."""
    STAGE1_HEAD_ONLY = 1       # Train only face-specific head
    STAGE2_PARTIAL_FINETUNE = 2  # Unfreeze last 4 backbone blocks
    STAGE3_FULL_FINETUNE = 3     # Full model fine-tuning


@dataclass
class TransferModelConfig:
    """Configuration for transfer learning model."""

    # Backbone settings
    backbone_blocks: int = 16  # Number of RRDB blocks to use from backbone
    freeze_blocks: int = 16    # Number of blocks to freeze initially

    # Face-specific head
    head_blocks: int = 4       # Number of RCAB blocks in face head
    head_channels: int = 64    # Channels in face head

    # Training
    scale_factor: int = 4

    # Stage-specific learning rates
    stage1_lr: float = 2e-4
    stage2_lr: float = 2e-5
    stage3_lr: float = 1e-5


class FaceSpecificHead(nn.Module):
    """
    Face-specific processing head.

    Uses RCAB blocks optimized for facial features.
    """

    def __init__(self,
                 in_channels: int = 64,
                 num_blocks: int = 4,
                 scale_factor: int = 4):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_blocks: Number of RCAB blocks
            scale_factor: Upscaling factor
        """
        super().__init__()

        # RCAB blocks for face-specific processing
        self.rcab_blocks = nn.ModuleList([
            RCAB(num_channels=in_channels, reduction_ratio=4)
            for _ in range(num_blocks)
        ])

        # Conv after RCAB blocks
        self.conv_after = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # Upsampling
        self.upsample = UpsampleModule(in_channels, scale_factor)

        # Final reconstruction
        self.conv_last = nn.Conv2d(in_channels, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        for rcab in self.rcab_blocks:
            x = rcab(x)

        x = self.conv_after(x)
        x = x + residual

        x = self.upsample(x)
        x = self.conv_last(x)

        return x


class TransferSRModel(nn.Module):
    """
    Transfer Learning Super-Resolution Model.

    Uses pre-trained ESRGAN backbone with trainable face-specific head.
    Supports progressive unfreezing for fine-tuning.
    """

    def __init__(self,
                 config: Optional[TransferModelConfig] = None,
                 pretrained_path: Optional[str] = None):
        """
        Args:
            config: Model configuration
            pretrained_path: Path to pre-trained backbone weights
        """
        super().__init__()

        if config is None:
            config = TransferModelConfig()
        self.config = config

        # Create backbone (ESRGAN-style)
        self.backbone = self._create_backbone()

        # Create face-specific head
        self.face_head = FaceSpecificHead(
            in_channels=config.head_channels,
            num_blocks=config.head_blocks,
            scale_factor=config.scale_factor
        )

        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Initialize training stage
        self.current_stage = TrainingStage.STAGE1_HEAD_ONLY
        self._update_frozen_layers()

    def _create_backbone(self) -> nn.Module:
        """Create the backbone feature extractor."""
        # Initial conv
        conv_first = nn.Conv2d(3, self.config.head_channels, 3, 1, 1)

        # RRDB blocks (from ESRGAN)
        body = nn.ModuleList([
            RRDB(self.config.head_channels, num_grow_ch=32)
            for _ in range(self.config.backbone_blocks)
        ])

        # Conv after body
        conv_body = nn.Conv2d(
            self.config.head_channels, self.config.head_channels, 3, 1, 1
        )

        return nn.ModuleDict({
            'conv_first': conv_first,
            'body': body,
            'conv_body': conv_body
        })

    def _load_pretrained(self, path: str) -> None:
        """Load pre-trained backbone weights."""
        state_dict = torch.load(path, map_location='cpu')

        # Handle different formats
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']

        # Map weights to backbone
        backbone_dict = {}
        for key, value in state_dict.items():
            if key.startswith('conv_first'):
                backbone_dict[f'conv_first.{key.split(".")[-1]}'] = value
            elif key.startswith('body.'):
                parts = key.split('.')
                block_idx = int(parts[1])
                if block_idx < self.config.backbone_blocks:
                    new_key = '.'.join(['body', str(block_idx)] + parts[2:])
                    backbone_dict[new_key] = value
            elif key.startswith('conv_body'):
                backbone_dict[f'conv_body.{key.split(".")[-1]}'] = value

        # Load with strict=False to allow missing keys
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"Loaded pre-trained weights from {path}")

    def set_training_stage(self, stage: TrainingStage) -> None:
        """
        Set the current training stage.

        Args:
            stage: Training stage to set
        """
        self.current_stage = stage
        self._update_frozen_layers()
        print(f"Training stage set to: {stage.name}")

    def _update_frozen_layers(self) -> None:
        """Update which layers are frozen based on current stage."""
        if self.current_stage == TrainingStage.STAGE1_HEAD_ONLY:
            # Freeze entire backbone
            self._freeze_backbone(0, self.config.backbone_blocks)
            self._unfreeze_head()

        elif self.current_stage == TrainingStage.STAGE2_PARTIAL_FINETUNE:
            # Freeze first N-4 blocks, unfreeze last 4
            freeze_until = self.config.backbone_blocks - 4
            self._freeze_backbone(0, freeze_until)
            self._unfreeze_backbone(freeze_until, self.config.backbone_blocks)
            self._unfreeze_head()

        elif self.current_stage == TrainingStage.STAGE3_FULL_FINETUNE:
            # Unfreeze everything
            self._unfreeze_backbone(0, self.config.backbone_blocks)
            self._unfreeze_head()

    def _freeze_backbone(self, start: int, end: int) -> None:
        """Freeze backbone blocks from start to end."""
        # Freeze conv_first
        for param in self.backbone['conv_first'].parameters():
            param.requires_grad = False

        # Freeze specified body blocks
        for i in range(start, min(end, len(self.backbone['body']))):
            for param in self.backbone['body'][i].parameters():
                param.requires_grad = False

        # Freeze conv_body if freezing all
        if end >= self.config.backbone_blocks:
            for param in self.backbone['conv_body'].parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self, start: int, end: int) -> None:
        """Unfreeze backbone blocks from start to end."""
        for i in range(start, min(end, len(self.backbone['body']))):
            for param in self.backbone['body'][i].parameters():
                param.requires_grad = True

        # Unfreeze conv_body when unfreezing last blocks
        if end >= self.config.backbone_blocks:
            for param in self.backbone['conv_body'].parameters():
                param.requires_grad = True

    def _unfreeze_head(self) -> None:
        """Unfreeze face-specific head."""
        for param in self.face_head.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input LR image (B, 3, H, W)

        Returns:
            SR image (B, 3, H*scale, W*scale)
        """
        # Backbone feature extraction
        feat = self.backbone['conv_first'](x)
        body_feat = feat

        for block in self.backbone['body']:
            body_feat = block(body_feat)

        body_feat = self.backbone['conv_body'](body_feat)
        feat = feat + body_feat

        # Face-specific head
        out = self.face_head(feat)

        return out

    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer.

        Returns different LR for backbone vs head.
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'face_head' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        # Get appropriate learning rates
        if self.current_stage == TrainingStage.STAGE1_HEAD_ONLY:
            head_lr = self.config.stage1_lr
            backbone_lr = 0  # Not used, backbone frozen
        elif self.current_stage == TrainingStage.STAGE2_PARTIAL_FINETUNE:
            head_lr = self.config.stage2_lr
            backbone_lr = self.config.stage2_lr * 0.1
        else:  # STAGE3
            head_lr = self.config.stage3_lr
            backbone_lr = self.config.stage3_lr

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': head_lr})

        return param_groups

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'name': 'TransferSRModel',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 ** 2),
            'backbone_blocks': self.config.backbone_blocks,
            'head_blocks': self.config.head_blocks,
            'current_stage': self.current_stage.name,
            'frozen_params': total_params - trainable_params,
        }


def create_transfer_model(
    pretrained_path: Optional[str] = None,
    **kwargs
) -> TransferSRModel:
    """
    Factory function to create transfer learning model.

    Args:
        pretrained_path: Path to pre-trained weights
        **kwargs: Config overrides

    Returns:
        TransferSRModel
    """
    config = TransferModelConfig(**kwargs)
    return TransferSRModel(config, pretrained_path)


if __name__ == '__main__':
    # Test the model
    print("Creating Transfer Learning model...")
    model = create_transfer_model()

    info = model.get_model_info()
    print(f"\nModel Info (Stage 1 - Head Only):")
    for k, v in info.items():
        if isinstance(v, int) and v > 1000:
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v}")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test stage transition
    print("\nTesting stage transition...")
    model.set_training_stage(TrainingStage.STAGE2_PARTIAL_FINETUNE)
    info = model.get_model_info()
    print(f"Stage 2 trainable params: {info['trainable_params']:,}")

    model.set_training_stage(TrainingStage.STAGE3_FULL_FINETUNE)
    info = model.get_model_info()
    print(f"Stage 3 trainable params: {info['trainable_params']:,}")

    print("\nSuccess!")
