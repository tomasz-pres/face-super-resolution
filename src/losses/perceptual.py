"""
Perceptual Loss using VGG19 Features
=====================================
Computes perceptual similarity using pre-trained VGG19 features.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional, Dict


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.

    Extracts features from specified layers of pre-trained VGG19.
    """

    # Layer name to index mapping for VGG19
    LAYER_MAP = {
        'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
        'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
        'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13,
        'conv3_3': 14, 'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
        'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22,
        'conv4_3': 23, 'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
        'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31,
        'conv5_3': 32, 'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36,
    }

    def __init__(self,
                 layers: List[str] = ['conv3_4', 'conv4_4'],
                 normalize: bool = True,
                 requires_grad: bool = False):
        """
        Args:
            layers: List of layer names to extract features from
            normalize: Whether to normalize input with ImageNet stats
            requires_grad: Whether to compute gradients for VGG weights
        """
        super().__init__()

        self.layers = layers
        self.normalize = normalize

        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg_features = vgg.features

        # Get max layer index needed
        max_idx = max(self.LAYER_MAP.get(layer, 0) for layer in layers)

        # Extract features up to max layer
        self.features = nn.Sequential(*list(vgg_features.children())[:max_idx + 1])

        # Store layer indices
        self.layer_indices = [self.LAYER_MAP[layer] for layer in layers]

        # Freeze weights and set to eval mode
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()  # Critical: keeps batchnorm/dropout in eval mode

        # ImageNet normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True):
        """Override train to keep VGG in eval mode."""
        # Always keep VGG features in eval mode
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers.

        Args:
            x: Input tensor (B, 3, H, W) in range [0, 1]

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Ensure eval mode
        self.features.eval()

        # Normalize input
        if self.normalize:
            x = (x - self.mean) / self.std

        # Extract features
        features = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_indices:
                layer_name = self.layers[self.layer_indices.index(idx)]
                features[layer_name] = x

        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.

    Computes L1/L2 distance between VGG features of predicted and target images.
    """

    def __init__(self,
                 layers: List[str] = ['conv3_4', 'conv4_4'],
                 weights: Optional[Dict[str, float]] = None,
                 criterion: str = 'l1',
                 normalize: bool = True):
        """
        Args:
            layers: VGG layers to use for loss computation
            weights: Per-layer weights (default: equal weights)
            criterion: 'l1' or 'l2' distance
            normalize: Whether to normalize VGG input
        """
        super().__init__()

        self.feature_extractor = VGGFeatureExtractor(
            layers=layers,
            normalize=normalize,
            requires_grad=False
        )

        self.layers = layers
        self.weights = weights or {layer: 1.0 for layer in layers}

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]

        Returns:
            Perceptual loss value
        """
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        # Compute weighted loss
        loss = 0.0
        for layer in self.layers:
            weight = self.weights.get(layer, 1.0)
            loss += weight * self.criterion(
                pred_features[layer],
                target_features[layer]
            )

        return loss


if __name__ == '__main__':
    # Test perceptual loss
    print("Testing Perceptual Loss...")

    loss_fn = PerceptualLoss()

    pred = torch.randn(2, 3, 256, 256).clamp(0, 1)
    target = torch.randn(2, 3, 256, 256).clamp(0, 1)

    loss = loss_fn(pred, target)
    print(f"Perceptual loss: {loss.item():.4f}")

    # Test gradient flow
    pred.requires_grad = True
    loss = loss_fn(pred, target)
    loss.backward()
    print(f"Gradient computed: {pred.grad is not None}")

    print("Success!")
