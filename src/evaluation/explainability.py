"""
Model Explainability with GradCAM
==================================
Visualize where the model focuses attention during super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Super-Resolution.

    Adapted for regression tasks (no class labels).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The model to explain
            target_layer: The layer to compute CAM for
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor,
                 target_region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_region: Optional (x, y, w, h) region to focus on

        Returns:
            GradCAM heatmap as numpy array
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # For SR, use mean of output as target (or specific region)
        if target_region is not None:
            x, y, w, h = target_region
            target = output[:, :, y:y+h, x:x+w].mean()
        else:
            target = output.mean()

        # Backward pass
        self.model.zero_grad()
        target.backward()

        # Compute GradCAM
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class AttentionExtractor:
    """
    Extract attention weights from channel attention modules.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model with channel attention modules
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self):
        """Register hooks on channel attention modules."""
        for name, module in self.model.named_modules():
            if 'channel_attention' in name.lower():
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self._save_attention(n, m, i, o)
                )
                self.hooks.append(hook)

    def _save_attention(self, name: str, module: nn.Module,
                        input: tuple, output: torch.Tensor):
        """Save attention weights from forward pass."""
        # The attention module outputs scaled features
        # We need to extract the attention weights before scaling
        with torch.no_grad():
            x = input[0]
            b, c, _, _ = x.size()

            # Compute attention weights (assuming SE-style attention)
            if hasattr(module, 'global_pool') and hasattr(module, 'fc'):
                y = module.global_pool(x).view(b, c)
                weights = module.fc(y)
                self.attention_weights[name] = weights.cpu().numpy()

    def extract(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract attention weights for an input.

        Args:
            input_tensor: Input image tensor

        Returns:
            Dictionary of attention weights per module
        """
        self.attention_weights = {}
        self.register_hooks()

        with torch.no_grad():
            _ = self.model(input_tensor)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        return self.attention_weights


def visualize_gradcam(input_image: np.ndarray,
                      cam_heatmap: np.ndarray,
                      output_image: np.ndarray,
                      alpha: float = 0.4,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay GradCAM heatmap on images.

    Args:
        input_image: Original input image (H, W, 3)
        cam_heatmap: GradCAM heatmap (h, w)
        output_image: Model output image (H, W, 3)
        alpha: Overlay transparency
        colormap: OpenCV colormap

    Returns:
        Overlay visualization
    """
    # Resize heatmap to match output size
    heatmap_resized = cv2.resize(cam_heatmap, (output_image.shape[1], output_image.shape[0]))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on output
    overlay = (1 - alpha) * output_image.astype(float) + alpha * heatmap_colored.astype(float)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def create_explainability_figure(input_image: np.ndarray,
                                 output_image: np.ndarray,
                                 cam_heatmap: np.ndarray,
                                 attention_weights: Optional[Dict[str, np.ndarray]] = None,
                                 title: str = "Model Explainability",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive explainability figure.

    Args:
        input_image: Input LR image
        output_image: Output SR image
        cam_heatmap: GradCAM heatmap
        attention_weights: Optional channel attention weights
        title: Figure title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_cols = 4 if attention_weights else 3

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Input image
    axes[0].imshow(cv2.resize(input_image, (256, 256), interpolation=cv2.INTER_NEAREST))
    axes[0].set_title('Input (64x64)')
    axes[0].axis('off')

    # GradCAM heatmap
    heatmap_resized = cv2.resize(cam_heatmap, (256, 256))
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')

    # Overlay
    overlay = visualize_gradcam(input_image, cam_heatmap, output_image)
    axes[2].imshow(overlay)
    axes[2].set_title('Output with Attention')
    axes[2].axis('off')

    # Attention weights (if provided)
    if attention_weights and n_cols > 3:
        # Stack all attention weights
        all_weights = np.concatenate([w.flatten() for w in attention_weights.values()])
        axes[3].bar(range(len(all_weights)), all_weights, alpha=0.7)
        axes[3].set_title('Channel Attention Weights')
        axes[3].set_xlabel('Channel')
        axes[3].set_ylabel('Weight')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def generate_explainability_report(model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   output_dir: str,
                                   num_samples: int = 3,
                                   device: str = 'cuda') -> None:
    """
    Generate explainability visualizations for multiple samples.

    Args:
        model: Model to explain
        dataloader: Data loader
        output_dir: Output directory
        num_samples: Number of samples to visualize
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Find target layer (last conv before upsampling)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print("Could not find suitable target layer for GradCAM")
        return

    gradcam = GradCAM(model, target_layer)
    attention_extractor = AttentionExtractor(model)

    # Get samples
    batch = next(iter(dataloader))
    lr_batch = batch['lr'][:num_samples].to(device)
    hr_batch = batch['hr'][:num_samples].to(device)

    print(f"Generating explainability visualizations for {num_samples} samples...")

    for idx in range(num_samples):
        lr = lr_batch[idx:idx+1]
        hr = hr_batch[idx:idx+1]

        # Forward pass for output
        with torch.no_grad():
            sr = model(lr)
            sr = torch.clamp(sr, 0, 1)

        # GradCAM
        lr_grad = lr.clone().requires_grad_(True)
        cam = gradcam(lr_grad)

        # Attention weights
        attention = attention_extractor.extract(lr)

        # Convert to numpy
        lr_np = (lr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Create figure
        fig = create_explainability_figure(
            lr_np, sr_np, cam, attention,
            title=f"Sample {idx + 1} Explainability",
            save_path=str(output_dir / f'explainability_{idx + 1}.png')
        )
        plt.close(fig)

    gradcam.remove_hooks()
    print(f"Explainability report saved to {output_dir}")


if __name__ == '__main__':
    # Test
    print("Testing GradCAM...")

    from src.models import create_face_enhance_net

    model = create_face_enhance_net()

    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if 'conv_after_body' in name:
            target_layer = module
            break

    if target_layer is None:
        # Use last RCAB
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

    print(f"Target layer: {target_layer}")

    gradcam = GradCAM(model, target_layer)

    # Test input
    x = torch.randn(1, 3, 64, 64)
    cam = gradcam(x)

    print(f"GradCAM shape: {cam.shape}")
    print(f"GradCAM range: [{cam.min():.3f}, {cam.max():.3f}]")

    gradcam.remove_hooks()
    print("Success!")
