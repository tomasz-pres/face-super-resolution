"""
GradCAM for Super-Resolution Models
====================================
Visualizes which input regions contribute most to the super-resolved output.

For super-resolution (no classification), we compute gradients w.r.t.:
- Mean of output pixels (overall contribution)
- Specific output regions (localized analysis)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union


class GradCAM:
    """
    GradCAM for super-resolution models.

    Hooks into intermediate layers to capture activations and gradients,
    then computes attention heatmaps showing input region importance.
    """

    def __init__(self, model: torch.nn.Module, target_layers: List[str] = None):
        """
        Initialize GradCAM.

        Args:
            model: The super-resolution model
            target_layers: List of layer names to visualize.
                          If None, uses last residual group.
        """
        self.model = model
        self.model.eval()

        self.activations = {}
        self.gradients = {}
        self.hooks = []

        # Default target layers for FaceEnhanceNet
        if target_layers is None:
            target_layers = ['residual_groups']

        self.target_layers = target_layers
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Find and hook target layers
        for name, module in self.model.named_modules():
            for target in self.target_layers:
                if target in name:
                    self.hooks.append(
                        module.register_forward_hook(get_activation(name))
                    )
                    self.hooks.append(
                        module.register_full_backward_hook(get_gradient(name))
                    )

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_region: str = 'full',
        target_coords: Tuple[int, int, int, int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Input LR image tensor [1, 3, H, W]
            target_region: 'full' (whole image), 'center', 'eyes', 'mouth', or 'custom'
            target_coords: For 'custom' region: (x1, y1, x2, y2) in output space

        Returns:
            Tuple of (heatmap, super_resolved_image) as numpy arrays
        """
        self.model.zero_grad()

        # Enable gradients for input
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Define target for backprop based on region
        if target_region == 'full':
            # Use mean of all output pixels
            target = output.mean()
        elif target_region == 'center':
            # Center 64x64 region of 256x256 output
            h, w = output.shape[2], output.shape[3]
            ch, cw = h // 2, w // 2
            target = output[:, :, ch-32:ch+32, cw-32:cw+32].mean()
        elif target_region == 'eyes':
            # Upper-middle region (approximate eye area)
            h, w = output.shape[2], output.shape[3]
            target = output[:, :, h//4:h//2, w//4:3*w//4].mean()
        elif target_region == 'mouth':
            # Lower-middle region (approximate mouth area)
            h, w = output.shape[2], output.shape[3]
            target = output[:, :, 5*h//8:7*h//8, w//4:3*w//4].mean()
        elif target_region == 'custom' and target_coords is not None:
            x1, y1, x2, y2 = target_coords
            target = output[:, :, y1:y2, x1:x2].mean()
        else:
            target = output.mean()

        # Backward pass
        target.backward()

        # Get the last captured activation and gradient
        if not self.activations or not self.gradients:
            raise RuntimeError("No activations/gradients captured. Check target_layers.")

        # Use the deepest layer
        layer_name = list(self.activations.keys())[-1]
        activations = self.activations[layer_name]
        gradients = self.gradients[layer_name]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU to keep positive contributions
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        sr_image = output.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        sr_image = np.clip(sr_image * 255, 0, 255).astype(np.uint8)

        return heatmap, sr_image

    def generate_multi_layer_cam(
        self,
        input_tensor: torch.Tensor
    ) -> dict:
        """
        Generate CAM for all captured layers.

        Returns:
            Dict mapping layer names to heatmaps
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)
        output.mean().backward()

        cams = {}
        for layer_name in self.activations.keys():
            activations = self.activations[layer_name]
            gradients = self.gradients[layer_name]

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            cam = F.interpolate(
                cam,
                size=input_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            cams[layer_name] = cam.squeeze().cpu().numpy()

        return cams


def apply_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        image: Original image (H, W, 3) RGB uint8
        heatmap: Attention heatmap (H, W) float [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Blended image with heatmap overlay
    """
    # Ensure contiguous arrays
    image = np.ascontiguousarray(image)
    heatmap = np.ascontiguousarray(heatmap)

    # Resize heatmap to match image if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to color
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return np.ascontiguousarray(blended)


def create_gradcam_visualization(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    hr_image: np.ndarray = None,
    target_layers: List[str] = None,
    device: str = 'cuda'
) -> dict:
    """
    Create comprehensive GradCAM visualization.

    Args:
        model: Super-resolution model
        lr_image: Low-resolution input (H, W, 3) RGB uint8
        hr_image: Optional high-resolution ground truth
        target_layers: Layers to visualize
        device: Computation device

    Returns:
        Dict with visualization images and data
    """
    if target_layers is None:
        target_layers = ['residual_groups', 'channel_attention', 'conv_last']

    # Prepare input
    lr_tensor = torch.from_numpy(lr_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_tensor = lr_tensor.to(device)

    # Initialize GradCAM
    gradcam = GradCAM(model, target_layers)

    results = {
        'lr_image': lr_image,
        'hr_image': hr_image,
        'heatmaps': {},
        'overlays': {},
    }

    # Generate CAM for different regions
    regions = ['full', 'center', 'eyes', 'mouth']

    for region in regions:
        try:
            heatmap, sr_image = gradcam.generate_cam(lr_tensor, target_region=region)
            results['heatmaps'][region] = heatmap

            # Create overlay on LR image (upscaled for display)
            lr_upscaled = cv2.resize(lr_image, (256, 256), interpolation=cv2.INTER_NEAREST)
            heatmap_upscaled = cv2.resize(heatmap, (256, 256))
            overlay = apply_heatmap(lr_upscaled, heatmap_upscaled, alpha=0.5)
            results['overlays'][region] = overlay

            if region == 'full':
                results['sr_image'] = sr_image
        except Exception as e:
            print(f"Warning: Could not generate CAM for region '{region}': {e}")

    # Clean up
    gradcam.remove_hooks()

    return results


def visualize_attention_flow(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Visualize attention flow through the network layers.

    Creates a multi-panel visualization showing how attention
    evolves through different network depths.

    Args:
        model: Super-resolution model
        lr_image: Input image (H, W, 3) RGB uint8
        device: Computation device

    Returns:
        Visualization image showing attention at different layers
    """
    # Target specific layers at different depths
    layer_targets = [
        'residual_groups.0',  # Early
        'residual_groups.1',  # Middle (if exists)
        'residual_groups.2',  # Late (if exists)
    ]

    lr_tensor = torch.from_numpy(lr_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_tensor = lr_tensor.to(device)

    panels = []

    # LR input (upscaled for display)
    lr_display = cv2.resize(lr_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    lr_display = np.ascontiguousarray(lr_display)
    cv2.putText(lr_display, "LR Input", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(lr_display)

    # Generate CAM for each layer depth
    for i, target in enumerate(layer_targets):
        try:
            gradcam = GradCAM(model, [target])
            heatmap, sr_image = gradcam.generate_cam(lr_tensor)
            gradcam.remove_hooks()

            # Overlay on LR
            heatmap_up = cv2.resize(heatmap, (256, 256))
            overlay = apply_heatmap(lr_display.copy(), heatmap_up, alpha=0.6)
            overlay = np.ascontiguousarray(overlay)

            label = f"Layer {i+1}"
            cv2.putText(overlay, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels.append(overlay)
        except:
            pass

    # SR output
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_out = sr_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    sr_out = np.clip(sr_out * 255, 0, 255).astype(np.uint8)
    sr_out = np.ascontiguousarray(sr_out)
    cv2.putText(sr_out, "SR Output", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(sr_out)

    # Combine panels
    visualization = np.hstack(panels)

    return visualization
