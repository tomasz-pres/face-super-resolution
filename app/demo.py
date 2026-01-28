"""
Gradio Demo for Face Super-Resolution

Interactive web interface to test and compare face super-resolution models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from typing import Tuple, Optional, Dict
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.models import create_face_enhance_net


# Global models (loaded once)
_models = {}
_lpips_model = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model configuration from checkpoint weights."""
    config = {
        'num_channels': 64,
        'num_groups': 3,
        'blocks_per_group': 4,
        'reduction_ratio': 4,
        'scale_factor': 4,
        'res_scale': 0.2,
    }

    # Count residual groups
    group_indices = set()
    for key in state_dict.keys():
        if key.startswith('residual_groups.') and '.blocks.' in key:
            group_idx = int(key.split('.')[1])
            group_indices.add(group_idx)
    if group_indices:
        config['num_groups'] = max(group_indices) + 1

    # Count blocks per group
    block_indices = set()
    for key in state_dict.keys():
        if 'residual_groups.0.blocks.' in key:
            parts = key.split('.')
            block_idx_pos = parts.index('blocks') + 1
            if block_idx_pos < len(parts):
                try:
                    block_idx = int(parts[block_idx_pos])
                    block_indices.add(block_idx)
                except ValueError:
                    pass
    if block_indices:
        config['blocks_per_group'] = max(block_indices) + 1

    # Get num_channels from conv_first
    if 'conv_first.weight' in state_dict:
        config['num_channels'] = state_dict['conv_first.weight'].shape[0]

    # Get reduction_ratio from channel attention
    ca_key = 'residual_groups.0.blocks.0.channel_attention.fc.0.weight'
    if ca_key in state_dict:
        reduced_channels = state_dict[ca_key].shape[0]
        config['reduction_ratio'] = config['num_channels'] // reduced_channels

    return config


def load_model(checkpoint_path: str, device: str = None) -> torch.nn.Module:
    """Load a model from checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Infer config
    model_config = infer_model_config_from_state_dict(state_dict)

    # Create and load model
    model = create_face_enhance_net(**model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def get_lpips_model(device: str = None):
    """Get or create LPIPS model."""
    global _lpips_model
    if device is None:
        device = get_device()
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
        _lpips_model.eval()
    return _lpips_model


def load_models_from_checkpoints():
    """Load all available model checkpoints."""
    global _models
    device = get_device()

    checkpoint_dir = project_root / 'checkpoints'

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    # Load ALL .pth files in checkpoints folder
    checkpoint_files = list(checkpoint_dir.glob('*.pth'))

    for checkpoint_path in checkpoint_files:
        filename = checkpoint_path.name
        # Create a readable name from filename
        name = filename.replace('.pth', '').replace('_', ' ').title()

        try:
            print(f"Loading '{name}' from {filename}...")
            _models[name] = load_model(str(checkpoint_path), device)
            print(f"  Loaded successfully!")
        except Exception as e:
            print(f"  Failed to load {filename}: {e}")

    return list(_models.keys())


def compute_metrics(sr_image: np.ndarray, hr_image: np.ndarray, device: str = None) -> Dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS."""
    if device is None:
        device = get_device()

    # Ensure same size
    if sr_image.shape != hr_image.shape:
        sr_image = cv2.resize(sr_image, (hr_image.shape[1], hr_image.shape[0]))

    psnr_val = psnr(hr_image, sr_image, data_range=255)
    ssim_val = ssim(hr_image, sr_image, data_range=255, channel_axis=2)

    # LPIPS
    lpips_model = get_lpips_model(device)
    sr_tensor = torch.from_numpy(sr_image).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    hr_tensor = torch.from_numpy(hr_image).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    sr_tensor = sr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)

    with torch.no_grad():
        lpips_val = lpips_model(sr_tensor, hr_tensor).item()

    return {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}


def generate_lr(hr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """Generate LR image using F.interpolate (bicubic) to match training."""
    # Convert to tensor for F.interpolate (BGR to RGB)
    hr_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    hr_tensor = torch.from_numpy(hr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # Downscale using F.interpolate (same as trainer)
    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_tensor = F.interpolate(hr_tensor, size=(lr_h, lr_w), mode='bicubic', align_corners=False)

    # Convert back to numpy BGR
    lr_np = lr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    lr_np = np.clip(lr_np * 255, 0, 255).astype(np.uint8)
    lr_image = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)

    return lr_image


def upscale_opencv(lr_image: np.ndarray, target_size: Tuple[int, int], method: int) -> np.ndarray:
    """Upscale using OpenCV interpolation."""
    return cv2.resize(lr_image, target_size, interpolation=method)


def upscale_model(lr_image: np.ndarray, model: torch.nn.Module, device: str = None) -> np.ndarray:
    """Upscale using our trained model."""
    if device is None:
        device = get_device()

    # Convert BGR to RGB and normalize
    lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    lr_tensor = torch.from_numpy(lr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_tensor = lr_tensor.to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Convert back
    sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)
    sr_rgb = (sr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

    return sr_bgr


def process_image(
    input_image: np.ndarray,
    model_name: str,
    show_bicubic: bool = True,
    show_lanczos: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Process an input image through super-resolution.

    Returns: (lr_display, sr_output, comparison_grid, metrics_text)
    """
    device = get_device()

    if input_image is None:
        return None, None, None, "Please upload an image"

    # Convert from RGB (Gradio) to BGR (OpenCV)
    input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Determine if input is LR or HR based on size
    h, w = input_bgr.shape[:2]

    if max(h, w) <= 128:
        # Input is already LR, upscale to get SR
        lr_image = input_bgr
        hr_image = None  # No ground truth
        target_size = (w * 4, h * 4)
    else:
        # Input is HR, center crop to 256x256 (better for webcam/face images)
        if h != 256 or w != 256:
            # Center crop to square first
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            input_bgr = input_bgr[top:top + min_dim, left:left + min_dim]
            # Then resize to 256x256
            input_bgr = cv2.resize(input_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        hr_image = input_bgr

        # Create LR (64x64) using F.interpolate bicubic (same as training)
        lr_image = generate_lr(hr_image, scale_factor=4)
        target_size = (256, 256)

    # Get model
    if model_name not in _models:
        return None, None, None, f"Model '{model_name}' not loaded"

    model = _models[model_name]

    # Upscale with our model
    sr_model = upscale_model(lr_image, model, device)

    # Upscale with OpenCV methods for comparison
    sr_bicubic = upscale_opencv(lr_image, target_size, cv2.INTER_CUBIC)
    sr_lanczos = upscale_opencv(lr_image, target_size, cv2.INTER_LANCZOS4)

    # Build metrics text
    metrics_lines = []
    metrics_lines.append("=" * 50)
    metrics_lines.append("SUPER-RESOLUTION METRICS")
    metrics_lines.append("=" * 50)

    if hr_image is not None:
        # Compute metrics against ground truth
        metrics_model = compute_metrics(cv2.cvtColor(sr_model, cv2.COLOR_BGR2RGB),
                                        cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB), device)
        metrics_bicubic = compute_metrics(cv2.cvtColor(sr_bicubic, cv2.COLOR_BGR2RGB),
                                          cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB), device)
        metrics_lanczos = compute_metrics(cv2.cvtColor(sr_lanczos, cv2.COLOR_BGR2RGB),
                                          cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB), device)

        metrics_lines.append(f"\n{'Method':<15} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
        metrics_lines.append("-" * 50)
        metrics_lines.append(f"{'Bicubic':<15} {metrics_bicubic['psnr']:>8.2f}dB {metrics_bicubic['ssim']:>10.4f} {metrics_bicubic['lpips']:>10.4f}")
        metrics_lines.append(f"{'Lanczos4':<15} {metrics_lanczos['psnr']:>8.2f}dB {metrics_lanczos['ssim']:>10.4f} {metrics_lanczos['lpips']:>10.4f}")
        metrics_lines.append(f"{'Our Model':<15} {metrics_model['psnr']:>8.2f}dB {metrics_model['ssim']:>10.4f} {metrics_model['lpips']:>10.4f}")
        metrics_lines.append("-" * 50)

        # Improvements
        psnr_imp = metrics_model['psnr'] - metrics_bicubic['psnr']
        ssim_imp = metrics_model['ssim'] - metrics_bicubic['ssim']
        lpips_imp = metrics_model['lpips'] - metrics_bicubic['lpips']  # Negative = better (lower)

        metrics_lines.append(f"\nImprovement vs Bicubic:")
        metrics_lines.append(f"  PSNR:  {psnr_imp:+.2f} dB")
        metrics_lines.append(f"  SSIM:  {ssim_imp:+.4f}")
        metrics_lines.append(f"  LPIPS: {lpips_imp:+.4f} (lower=better)")
    else:
        metrics_lines.append("\n(No ground truth - metrics unavailable)")
        metrics_lines.append("Upload a 256x256 image for full metrics")

    metrics_lines.append("=" * 50)
    metrics_text = "\n".join(metrics_lines)

    # Create comparison grid
    comparison_images = []
    labels = []

    # LR (upscaled for display)
    lr_display = cv2.resize(lr_image, target_size, interpolation=cv2.INTER_NEAREST)
    comparison_images.append(lr_display)
    labels.append("LR Input")

    if show_bicubic:
        comparison_images.append(sr_bicubic)
        labels.append("Bicubic")

    if show_lanczos:
        comparison_images.append(sr_lanczos)
        labels.append("Lanczos4")

    comparison_images.append(sr_model)
    labels.append("Our Model")

    if hr_image is not None:
        comparison_images.append(hr_image)
        labels.append("Ground Truth")

    # Add labels to images
    labeled_images = []
    for img, label in zip(comparison_images, labels):
        img_copy = img.copy()
        cv2.rectangle(img_copy, (0, 0), (img.shape[1], 25), (0, 0, 0), -1)
        cv2.putText(img_copy, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        labeled_images.append(img_copy)

    # Stack into grid
    comparison_grid = np.hstack(labeled_images)

    # Convert outputs to RGB for Gradio
    lr_display_rgb = cv2.cvtColor(lr_display, cv2.COLOR_BGR2RGB)
    sr_model_rgb = cv2.cvtColor(sr_model, cv2.COLOR_BGR2RGB)
    comparison_rgb = cv2.cvtColor(comparison_grid, cv2.COLOR_BGR2RGB)

    return lr_display_rgb, sr_model_rgb, comparison_rgb, metrics_text


def load_sample_image(sample_name: str) -> np.ndarray:
    """Load a sample image from validation set."""
    val_dir = project_root / 'data' / 'processed' / 'val' / 'HR'

    if not val_dir.exists():
        return None

    images = list(val_dir.glob('*.png'))
    if not images:
        return None

    # Get sample by index
    try:
        idx = int(sample_name.split('#')[1]) - 1
        if 0 <= idx < len(images):
            img = cv2.imread(str(images[idx]))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pass

    return None


def get_sample_options():
    """Get list of available sample images."""
    val_dir = project_root / 'data' / 'processed' / 'val' / 'HR'

    if not val_dir.exists():
        return ["No samples available"]

    images = list(val_dir.glob('*.png'))[:20]  # Limit to 20 samples
    return [f"Sample #{i+1}" for i in range(len(images))]


def create_demo():
    """Create the Gradio demo interface."""

    # Load models
    print("Loading models...")
    available_models = load_models_from_checkpoints()

    if not available_models:
        print("WARNING: No models found in checkpoints/")
        print("Please ensure checkpoint files exist (e.g., best_model.pth)")
        available_models = ["No models available"]

    # Get sample options
    sample_options = get_sample_options()

    # Build interface
    with gr.Blocks(title="Face Super-Resolution Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Face Super-Resolution Demo

        Upload a face image to enhance its resolution using our trained deep learning model.
        Compare results against traditional upscaling methods (Bicubic, Lanczos).

        **How to use:**
        1. Upload an image OR select a sample from the validation set
        2. Choose a model (if multiple are available)
        3. Click "Enhance" to process

        **Note:** For accurate metrics, upload a 256x256 image (will be downscaled to 64x64, then upscaled back)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input")
                input_image = gr.Image(label="Upload Face Image", type="numpy")

                with gr.Row():
                    sample_dropdown = gr.Dropdown(
                        choices=sample_options,
                        label="Or select a sample",
                        value=sample_options[0] if sample_options else None
                    )
                    load_sample_btn = gr.Button("Load Sample", size="sm")

                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Select Model"
                )

                with gr.Row():
                    show_bicubic = gr.Checkbox(label="Show Bicubic", value=True)
                    show_lanczos = gr.Checkbox(label="Show Lanczos", value=True)

                enhance_btn = gr.Button("Enhance", variant="primary", size="lg")

            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### Results")

                with gr.Row():
                    lr_output = gr.Image(label="Low Resolution Input", type="numpy")
                    sr_output = gr.Image(label="Super-Resolution Output", type="numpy")

                comparison_output = gr.Image(label="Comparison (scroll to see all)", type="numpy")

                metrics_output = gr.Textbox(
                    label="Metrics",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )

        # Examples section
        gr.Markdown("---")
        gr.Markdown("### About the Metrics")
        gr.Markdown("""
        | Metric | Description | Better |
        |--------|-------------|--------|
        | **PSNR** | Peak Signal-to-Noise Ratio - measures pixel-level accuracy | Higher |
        | **SSIM** | Structural Similarity - measures structural preservation | Higher |
        | **LPIPS** | Learned Perceptual Image Patch Similarity - measures perceptual quality | **Lower** |

        **Note:** GAN-trained models often have slightly lower PSNR/SSIM but much better LPIPS,
        because they generate sharper, more realistic details that may not exactly match the ground truth pixels.
        """)

        # Event handlers
        def on_load_sample(sample_name):
            img = load_sample_image(sample_name)
            return img

        load_sample_btn.click(
            fn=on_load_sample,
            inputs=[sample_dropdown],
            outputs=[input_image]
        )

        enhance_btn.click(
            fn=process_image,
            inputs=[input_image, model_dropdown, show_bicubic, show_lanczos],
            outputs=[lr_output, sr_output, comparison_output, metrics_output]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Face Super-Resolution Gradio Demo')
    parser.add_argument('--share', action='store_true', help='Create a public link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(share=args.share, server_port=args.port)
