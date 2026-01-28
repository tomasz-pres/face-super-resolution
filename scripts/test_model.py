#!/usr/bin/env python
"""
Test Script for Face Super-Resolution Model
============================================
Tests the trained FaceEnhanceNet model on HR images.

Usage:
    python scripts/test_model.py --checkpoint checkpoints/best_model.pth
    python scripts/test_model.py --checkpoint checkpoints/best_model.pth --image data/processed/val/HR/00001.png
    python scripts/test_model.py --checkpoint checkpoints/best_model.pth --num-images 10
    python scripts/test_model.py --checkpoint checkpoints/best_model.pth --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
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

from src.models import create_face_enhance_net


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

    return config


def load_model(checkpoint_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """Load the trained model with automatic config detection.

    Returns:
        tuple: (model, model_config) - the loaded model and its configuration
    """
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Get model config - priority: config file > inferred from weights
    if config_path and Path(config_path).exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        model_cfg = yaml_config.get('model', {}).get('custom', {})
        model_config = {
            'num_channels': model_cfg.get('num_channels', 64),
            'num_groups': model_cfg.get('num_groups', 3),
            'blocks_per_group': model_cfg.get('blocks_per_group', 4),
            'reduction_ratio': model_cfg.get('reduction_ratio', 4),
            'scale_factor': model_cfg.get('upscale_factor', 4),
            'res_scale': model_cfg.get('res_scale', 0.2),
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

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_psnr' in checkpoint:
        print(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")

    info = model.get_model_info()
    print(f"Model parameters: {info['total_params']:,}")

    return model, model_config


def generate_lr(hr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """Generate LR image using F.interpolate (bicubic) to match training."""
    # Convert to tensor for F.interpolate
    # BGR to RGB
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


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image (H, W, C) to tensor (1, C, H, W)."""
    # BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # HWC to CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)

    return tensor


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (1, C, H, W) to numpy image (H, W, C)."""
    image = tensor.squeeze(0).cpu().numpy()

    # CHW to HWC
    image = np.transpose(image, (1, 2, 0))

    # Clip and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def compute_metrics(sr_image: np.ndarray, hr_image: np.ndarray) -> dict:
    """Compute PSNR and SSIM metrics."""
    # Convert to RGB for metrics
    sr_rgb = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    hr_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

    psnr_val = psnr(hr_rgb, sr_rgb, data_range=255)
    ssim_val = ssim(hr_rgb, sr_rgb, data_range=255, channel_axis=2)

    return {'psnr': psnr_val, 'ssim': ssim_val}


def test_single_image(model, image_path: str, device: str = 'cuda',
                      save_output: bool = True, output_dir: str = 'outputs',
                      scale_factor: int = 4, verbose: bool = False):
    """Test model on a single image."""
    # Load HR image
    hr_image = cv2.imread(image_path)
    if hr_image is None:
        return None

    # Generate LR image
    lr_image = generate_lr(hr_image, scale_factor=scale_factor)

    # Convert to tensor
    lr_tensor = to_tensor(lr_image).to(device)

    # Super-resolve
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Convert back to numpy
    sr_image = to_numpy(sr_tensor)

    # Compute model metrics
    model_metrics = compute_metrics(sr_image, hr_image)

    # Compute bicubic baseline metrics
    bicubic_up = cv2.resize(lr_image, (hr_image.shape[1], hr_image.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
    bicubic_metrics = compute_metrics(bicubic_up, hr_image)

    # Save outputs
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem

        # Save SR image
        sr_path = output_path / f"{image_name}_sr.png"
        cv2.imwrite(str(sr_path), sr_image)

        # Save comparison (LR upscaled | SR | HR)
        comparison = np.hstack([bicubic_up, sr_image, hr_image])
        comp_path = output_path / f"{image_name}_comparison.png"
        cv2.imwrite(str(comp_path), comparison)

    return {
        'image': Path(image_path).name,
        'model_psnr': model_metrics['psnr'],
        'model_ssim': model_metrics['ssim'],
        'bicubic_psnr': bicubic_metrics['psnr'],
        'bicubic_ssim': bicubic_metrics['ssim'],
    }


def test_directory(model, hr_dir: str, device: str = 'cuda',
                   num_images: int = None, save_output: bool = True,
                   output_dir: str = 'outputs', scale_factor: int = 4):
    """Test model on all images in a directory."""
    hr_path = Path(hr_dir)

    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(hr_path.glob(ext))

    image_files = sorted(image_files)

    if num_images:
        image_files = image_files[:num_images]

    print(f"\nFound {len(image_files)} images in {hr_dir}\n")

    # Print header
    print("-" * 90)
    print(f"{'Image':<20} | {'Bicubic PSNR':>12} | {'Bicubic SSIM':>12} | {'Model PSNR':>12} | {'Model SSIM':>12}")
    print("-" * 90)

    results = []
    pbar = tqdm(image_files, desc="Testing", unit="img")

    for image_path in pbar:
        result = test_single_image(model, str(image_path), device, save_output, output_dir, scale_factor)
        if result:
            results.append(result)

            # Print metrics for this image
            tqdm.write(
                f"{result['image']:<20} | "
                f"{result['bicubic_psnr']:>10.2f} dB | "
                f"{result['bicubic_ssim']:>12.4f} | "
                f"{result['model_psnr']:>10.2f} dB | "
                f"{result['model_ssim']:>12.4f}"
            )

            # Update progress bar with running averages
            avg_model_psnr = np.mean([r['model_psnr'] for r in results])
            avg_model_ssim = np.mean([r['model_ssim'] for r in results])
            pbar.set_postfix({
                'avg_PSNR': f'{avg_model_psnr:.2f}',
                'avg_SSIM': f'{avg_model_ssim:.4f}'
            })

    print("-" * 90)

    # Print summary
    if results:
        avg_model_psnr = np.mean([r['model_psnr'] for r in results])
        avg_model_ssim = np.mean([r['model_ssim'] for r in results])
        avg_bicubic_psnr = np.mean([r['bicubic_psnr'] for r in results])
        avg_bicubic_ssim = np.mean([r['bicubic_ssim'] for r in results])

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Images tested: {len(results)}")
        print(f"\n{'Method':<20} | {'PSNR':>12} | {'SSIM':>12}")
        print("-" * 50)
        print(f"{'Bicubic':<20} | {avg_bicubic_psnr:>10.2f} dB | {avg_bicubic_ssim:>12.4f}")
        print(f"{'Model':<20} | {avg_model_psnr:>10.2f} dB | {avg_model_ssim:>12.4f}")
        print("-" * 50)
        print(f"{'Improvement':<20} | {avg_model_psnr - avg_bicubic_psnr:>+10.2f} dB | {avg_model_ssim - avg_bicubic_ssim:>+12.4f}")
        print("=" * 60)

        if save_output:
            print(f"\nOutputs saved to: {output_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Face Super-Resolution Model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (optional, will auto-detect from checkpoint)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image to test')
    parser.add_argument('--hr-dir', type=str, default='data/processed/val/HR',
                        help='Directory with HR images')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to test (default: all)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output images')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model (returns model and config)
    model, model_config = load_model(args.checkpoint, args.config, args.device)
    scale_factor = model_config.get('scale_factor', 4)

    # Test
    save_output = not args.no_save

    if args.image:
        # Test single image
        result = test_single_image(model, args.image, args.device, save_output, args.output_dir, scale_factor)
        if result:
            print(f"\n{'Method':<20} | {'PSNR':>12} | {'SSIM':>12}")
            print("-" * 50)
            print(f"{'Bicubic':<20} | {result['bicubic_psnr']:>10.2f} dB | {result['bicubic_ssim']:>12.4f}")
            print(f"{'Model':<20} | {result['model_psnr']:>10.2f} dB | {result['model_ssim']:>12.4f}")
            print("-" * 50)
            print(f"{'Improvement':<20} | {result['model_psnr'] - result['bicubic_psnr']:>+10.2f} dB | {result['model_ssim'] - result['bicubic_ssim']:>+12.4f}")
    else:
        # Test directory
        test_directory(model, args.hr_dir, args.device, args.num_images, save_output, args.output_dir, scale_factor)


if __name__ == '__main__':
    main()
