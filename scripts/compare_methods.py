#!/usr/bin/env python
"""
Compare Super-Resolution Methods
=================================
Compares trained model against standard OpenCV interpolation methods.

Methods compared:
- Nearest Neighbor (cv2.INTER_NEAREST)
- Bilinear (cv2.INTER_LINEAR)
- Bicubic (cv2.INTER_CUBIC)
- Lanczos4 (cv2.INTER_LANCZOS4)
- Our Model (FaceEnhanceNet)

Usage:
    python scripts/compare_methods.py --checkpoint checkpoints/best_model.pth
    python scripts/compare_methods.py --checkpoint checkpoints/best_model.pth --num-images 20
    python scripts/compare_methods.py --checkpoint checkpoints/best_model.pth --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import lpips

from src.models import create_face_enhance_net


# Global LPIPS model (loaded once)
_lpips_model = None


def get_lpips_model(device='cuda'):
    """Get or create LPIPS model (singleton pattern)."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
        _lpips_model.eval()
    return _lpips_model


# OpenCV interpolation methods to compare
OPENCV_METHODS = {
    'Nearest': cv2.INTER_NEAREST,
    'Bilinear': cv2.INTER_LINEAR,
    'Bicubic': cv2.INTER_CUBIC,
    'Lanczos4': cv2.INTER_LANCZOS4,
}


def get_model_config_from_checkpoint(checkpoint: dict) -> dict:
    """Extract model config from checkpoint if available."""
    config = {}

    # Try to get config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Map trainer config to model config if needed
        if isinstance(saved_config, dict):
            config = saved_config

    return config


def infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model configuration from state dict shapes."""
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
    block_counts = {}

    for key in state_dict.keys():
        if key.startswith('residual_groups.'):
            parts = key.split('.')
            group_idx = int(parts[1])
            group_indices.add(group_idx)

            if 'blocks.' in key:
                block_idx = int(parts[3])
                if group_idx not in block_counts:
                    block_counts[group_idx] = set()
                block_counts[group_idx].add(block_idx)

    if group_indices:
        config['num_groups'] = max(group_indices) + 1

    if block_counts:
        # Get blocks per group from first group
        config['blocks_per_group'] = max(block_counts.get(0, {0})) + 1

    # Infer num_channels from conv_first weight shape
    if 'conv_first.weight' in state_dict:
        config['num_channels'] = state_dict['conv_first.weight'].shape[0]

    # Infer reduction_ratio from channel attention FC layer
    for key in state_dict.keys():
        if 'channel_attention.fc.0.weight' in key:
            reduced_channels = state_dict[key].shape[0]
            config['reduction_ratio'] = config['num_channels'] // reduced_channels
            break

    # Check for res_scale in RCAB (look at the values, harder to infer)
    # Default to 0.2 for newer models, 1.0 for older

    return config


def load_model(checkpoint_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """Load the trained model with automatic config detection."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Get model config - priority: config file > checkpoint > inferred
    if config_path and Path(config_path).exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        model_config = yaml_config.get('model', {}).get('custom', {})
        model_config = {
            'num_channels': model_config.get('num_channels', 64),
            'num_groups': model_config.get('num_groups', 3),
            'blocks_per_group': model_config.get('blocks_per_group', 4),
            'reduction_ratio': model_config.get('reduction_ratio', 4),
            'scale_factor': model_config.get('upscale_factor', 4),
            'res_scale': model_config.get('res_scale', 0.2),
        }
    else:
        # Infer from state dict
        print("Inferring model config from checkpoint weights...")
        model_config = infer_model_config_from_state_dict(state_dict)

    print(f"Model config: {model_config}")

    # Create model
    model = create_face_enhance_net(**model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch {checkpoint['epoch']}")

    return model


def generate_lr(hr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """Generate LR image using F.interpolate to match training."""
    hr_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    hr_tensor = torch.from_numpy(hr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_tensor = F.interpolate(hr_tensor, size=(lr_h, lr_w), mode='bicubic', align_corners=False)

    lr_np = lr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    lr_np = np.clip(lr_np * 255, 0, 255).astype(np.uint8)
    lr_image = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)

    return lr_image


def upscale_opencv(lr_image: np.ndarray, target_size: Tuple[int, int], method: int) -> np.ndarray:
    """Upscale using OpenCV interpolation."""
    return cv2.resize(lr_image, target_size, interpolation=method)


def upscale_model(lr_image: np.ndarray, model, device: str = 'cuda') -> np.ndarray:
    """Upscale using trained model."""
    # BGR to RGB, normalize
    lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    lr_tensor = torch.from_numpy(lr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_tensor = lr_tensor.to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Back to numpy BGR
    sr_np = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
    sr_image = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

    return sr_image


def compute_metrics(sr_image: np.ndarray, hr_image: np.ndarray, device: str = 'cuda') -> Dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS."""
    sr_rgb = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    hr_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

    psnr_val = psnr(hr_rgb, sr_rgb, data_range=255)
    ssim_val = ssim(hr_rgb, sr_rgb, data_range=255, channel_axis=2)

    # Compute LPIPS (lower is better - more perceptually similar)
    lpips_model = get_lpips_model(device)

    # Convert to tensor: normalize to [-1, 1] as LPIPS expects
    sr_tensor = torch.from_numpy(sr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    hr_tensor = torch.from_numpy(hr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    sr_tensor = sr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)

    with torch.no_grad():
        lpips_val = lpips_model(sr_tensor, hr_tensor).item()

    return {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}


def compare_single_image(
    hr_path: str,
    model,
    device: str = 'cuda',
    save_comparison: bool = True,
    output_dir: str = 'outputs/comparisons'
) -> Dict[str, Dict[str, float]]:
    """Compare all methods on a single image."""

    hr_image = cv2.imread(hr_path)
    if hr_image is None:
        return None

    h, w = hr_image.shape[:2]
    target_size = (w, h)

    # Generate LR
    lr_image = generate_lr(hr_image, scale_factor=4)

    results = {}
    upscaled_images = {}

    # OpenCV methods
    for name, method in OPENCV_METHODS.items():
        sr = upscale_opencv(lr_image, target_size, method)
        metrics = compute_metrics(sr, hr_image, device)
        results[name] = metrics
        upscaled_images[name] = sr

    # Our model
    sr_model = upscale_model(lr_image, model, device)
    metrics_model = compute_metrics(sr_model, hr_image, device)
    results['Model'] = metrics_model
    upscaled_images['Model'] = sr_model

    # Save comparison image
    if save_comparison:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_name = Path(hr_path).stem

        # Create comparison grid: LR | Nearest | Bilinear | Bicubic | Lanczos4 | Model | HR
        lr_up = cv2.resize(lr_image, target_size, interpolation=cv2.INTER_NEAREST)

        # Add labels to images
        def add_label(img, label, psnr_val, ssim_val, lpips_val):
            img_copy = img.copy()
            # Add black bar at top for text
            cv2.rectangle(img_copy, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(img_copy, f"{label}", (5, 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img_copy, f"PSNR:{psnr_val:.1f} SSIM:{ssim_val:.3f}", (5, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(img_copy, f"LPIPS:{lpips_val:.3f} (lower=better)", (5, 37),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 255, 150), 1)
            return img_copy

        # Label each image
        labeled_images = []

        # LR (upscaled for display)
        lr_labeled = lr_up.copy()
        cv2.rectangle(lr_labeled, (0, 0), (lr_labeled.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(lr_labeled, "LR Input", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lr_labeled, f"64x64 -> 256x256", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        labeled_images.append(lr_labeled)

        # OpenCV methods
        for name in ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos4']:
            labeled = add_label(upscaled_images[name], name,
                              results[name]['psnr'], results[name]['ssim'], results[name]['lpips'])
            labeled_images.append(labeled)

        # Our model
        model_labeled = add_label(upscaled_images['Model'], "Our Model",
                                 results['Model']['psnr'], results['Model']['ssim'], results['Model']['lpips'])
        labeled_images.append(model_labeled)

        # HR (ground truth)
        hr_labeled = hr_image.copy()
        cv2.rectangle(hr_labeled, (0, 0), (hr_labeled.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(hr_labeled, "HR Ground Truth", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(hr_labeled, "256x256 original", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        labeled_images.append(hr_labeled)

        # Stack horizontally
        comparison = np.hstack(labeled_images)
        comp_path = output_path / f"{image_name}_comparison.png"
        cv2.imwrite(str(comp_path), comparison)

    return results


def run_comparison(
    model,
    hr_dir: str,
    device: str = 'cuda',
    num_images: int = None,
    save_comparisons: bool = True,
    output_dir: str = 'outputs/comparisons'
):
    """Run comparison on multiple images."""

    hr_path = Path(hr_dir)
    image_files = sorted(list(hr_path.glob('*.png')) + list(hr_path.glob('*.jpg')))

    if num_images:
        image_files = image_files[:num_images]

    print(f"\nComparing {len(image_files)} images...")
    print(f"Methods: {', '.join(list(OPENCV_METHODS.keys()) + ['Model'])}\n")

    # Aggregate results
    all_results = {name: {'psnr': [], 'ssim': [], 'lpips': []} for name in list(OPENCV_METHODS.keys()) + ['Model']}

    for img_path in tqdm(image_files, desc="Processing"):
        results = compare_single_image(str(img_path), model, device, save_comparisons, output_dir)
        if results:
            for method, metrics in results.items():
                all_results[method]['psnr'].append(metrics['psnr'])
                all_results[method]['ssim'].append(metrics['ssim'])
                all_results[method]['lpips'].append(metrics['lpips'])

    # Compute averages
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)
    print(f"\n{'Method':<15} | {'Avg PSNR':>12} | {'Avg SSIM':>12} | {'Avg LPIPS':>12} | {'PSNR vs Bicubic':>16}")
    print("-" * 85)

    bicubic_psnr = np.mean(all_results['Bicubic']['psnr'])

    # Sort by PSNR for display
    method_order = ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos4', 'Model']

    for method in method_order:
        avg_psnr = np.mean(all_results[method]['psnr'])
        avg_ssim = np.mean(all_results[method]['ssim'])
        avg_lpips = np.mean(all_results[method]['lpips'])
        diff = avg_psnr - bicubic_psnr
        diff_str = f"{diff:+.2f} dB" if method != 'Bicubic' else "baseline"

        # Highlight our model
        if method == 'Model':
            print("-" * 85)

        print(f"{method:<15} | {avg_psnr:>10.2f} dB | {avg_ssim:>12.4f} | {avg_lpips:>12.4f} | {diff_str:>16}")

    print("=" * 100)

    # Summary
    model_psnr = np.mean(all_results['Model']['psnr'])
    model_ssim = np.mean(all_results['Model']['ssim'])
    model_lpips = np.mean(all_results['Model']['lpips'])
    best_opencv_psnr = max(np.mean(all_results[m]['psnr']) for m in OPENCV_METHODS.keys())
    best_opencv_ssim = max(np.mean(all_results[m]['ssim']) for m in OPENCV_METHODS.keys())
    best_opencv_lpips = min(np.mean(all_results[m]['lpips']) for m in OPENCV_METHODS.keys())  # Lower is better

    print(f"\nOur Model vs Best OpenCV Method:")
    print(f"  PSNR:  {model_psnr:.2f} vs {best_opencv_psnr:.2f} ({model_psnr - best_opencv_psnr:+.2f} dB)")
    print(f"  SSIM:  {model_ssim:.4f} vs {best_opencv_ssim:.4f} ({model_ssim - best_opencv_ssim:+.4f})")
    print(f"  LPIPS: {model_lpips:.4f} vs {best_opencv_lpips:.4f} ({model_lpips - best_opencv_lpips:+.4f}) [lower is better]")

    if save_comparisons:
        print(f"\nComparison images saved to: {output_dir}/")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Compare SR methods')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (optional, will auto-detect from checkpoint)')
    parser.add_argument('--hr-dir', type=str, default='data/processed/val/HR',
                        help='Directory with HR images')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to compare')
    parser.add_argument('--output-dir', type=str, default='outputs/comparisons',
                        help='Output directory for comparison images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save comparison images')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    model = load_model(args.checkpoint, args.config, args.device)

    run_comparison(
        model=model,
        hr_dir=args.hr_dir,
        device=args.device,
        num_images=args.num_images,
        save_comparisons=not args.no_save,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
