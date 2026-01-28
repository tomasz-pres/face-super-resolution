#!/usr/bin/env python
"""
Compare All Super-Resolution Models
====================================
Automatically find and compare all models in checkpoints directory.

Usage:
    python scripts/compare_two_models.py
    python scripts/compare_two_models.py --num-images 50
    python scripts/compare_two_models.py --checkpoint-dir checkpoints
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import lpips

from src.models import create_face_enhance_net, create_transfer_model


# OpenCV interpolation methods for baseline comparison
OPENCV_METHODS = {
    'Bilinear': cv2.INTER_LINEAR,
    'Bicubic': cv2.INTER_CUBIC,
    'Lanczos4': cv2.INTER_LANCZOS4,
}

# Global LPIPS model
_lpips_model = None


def get_lpips_model(device='cuda'):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
        _lpips_model.eval()
    return _lpips_model


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

    if 'conv_first.weight' in state_dict:
        config['num_channels'] = state_dict['conv_first.weight'].shape[0]

    for key in state_dict.keys():
        if 'channel_attention.fc.0.weight' in key:
            reduced_channels = state_dict[key].shape[0]
            config['reduction_ratio'] = config['num_channels'] // reduced_channels
            break

    return config


def find_checkpoints(checkpoint_dir: str) -> dict:
    """Find all .pth files in checkpoint directory and create friendly names."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = {}

    for pth_file in checkpoint_path.glob('*.pth'):
        # Create friendly name from filename
        name = pth_file.stem

        # Map common names to display names
        if 'transfer' in name.lower():
            display_name = 'Transfer'
        elif 'custom' in name.lower() or 'final' in name.lower():
            display_name = 'Custom (GAN)'
        elif 'best' in name.lower():
            display_name = name.replace('_', ' ').title()
        else:
            display_name = name.replace('_', ' ').title()

        checkpoints[display_name] = str(pth_file)

    return checkpoints


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint with auto-detected config."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Detect model type from keys
    first_key = list(state_dict.keys())[0]

    if first_key.startswith('backbone.'):
        # TransferSRModel
        model = create_transfer_model()
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, {'type': 'transfer'}
    else:
        # FaceEnhanceNet
        model_config = infer_model_config_from_state_dict(state_dict)
        model = create_face_enhance_net(**model_config)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, model_config


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


def upscale_model(lr_image: np.ndarray, model, device: str = 'cuda') -> np.ndarray:
    """Upscale using trained model."""
    lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    lr_tensor = torch.from_numpy(lr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_tensor = lr_tensor.to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_np = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
    sr_image = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

    return sr_image


def compute_metrics(sr_image: np.ndarray, hr_image: np.ndarray, device: str = 'cuda') -> dict:
    """Compute PSNR, SSIM, and LPIPS."""
    sr_rgb = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    hr_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

    psnr_val = psnr(hr_rgb, sr_rgb, data_range=255)
    ssim_val = ssim(hr_rgb, sr_rgb, data_range=255, channel_axis=2)

    lpips_model = get_lpips_model(device)
    sr_tensor = torch.from_numpy(sr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    hr_tensor = torch.from_numpy(hr_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    sr_tensor = sr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)

    with torch.no_grad():
        lpips_val = lpips_model(sr_tensor, hr_tensor).item()

    return {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}


def main():
    parser = argparse.ArgumentParser(description='Compare all SR models in checkpoints directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--hr-dir', type=str, default='data/processed/test/HR',
                        help='Directory with HR test images')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to compare')
    parser.add_argument('--output-dir', type=str, default='outputs/model_comparison',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save comparison image every N images (0 to disable)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        print(f"No .pth files found in {args.checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} model(s):")
    for name, path in checkpoints.items():
        print(f"  - {name}: {path}")

    # Load all models
    models = {}
    for name, path in checkpoints.items():
        print(f"\nLoading {name}...")
        try:
            model, config = load_model(path, args.device)
            models[name] = model
            print(f"  Config: {config}")
        except Exception as e:
            print(f"  Failed to load: {e}")

    if not models:
        print("No models could be loaded!")
        return

    # Get images
    hr_path = Path(args.hr_dir)
    image_files = sorted(list(hr_path.glob('*.png')) + list(hr_path.glob('*.jpg')))[:args.num_images]

    print(f"\nComparing on {len(image_files)} test images...\n")
    print(f"Baseline methods: {', '.join(OPENCV_METHODS.keys())}")
    print(f"Trained models: {', '.join(models.keys())}\n")

    # Results storage - OpenCV methods + all models
    results = {}
    for method_name in OPENCV_METHODS.keys():
        results[method_name] = {'psnr': [], 'ssim': [], 'lpips': []}
    for name in models.keys():
        results[name] = {'psnr': [], 'ssim': [], 'lpips': []}

    # Process images
    for i, img_path in enumerate(tqdm(image_files, desc="Processing")):
        hr_image = cv2.imread(str(img_path))
        if hr_image is None:
            continue

        h, w = hr_image.shape[:2]
        target_size = (w, h)

        # Generate LR
        lr_image = generate_lr(hr_image, scale_factor=4)

        # Storage for this image
        sr_images = {}
        all_metrics = {}

        # OpenCV baseline methods
        for method_name, interpolation in OPENCV_METHODS.items():
            sr = cv2.resize(lr_image, target_size, interpolation=interpolation)
            metrics = compute_metrics(sr, hr_image, args.device)
            results[method_name]['psnr'].append(metrics['psnr'])
            results[method_name]['ssim'].append(metrics['ssim'])
            results[method_name]['lpips'].append(metrics['lpips'])
            sr_images[method_name] = sr
            all_metrics[method_name] = metrics

        # All trained models
        for name, model in models.items():
            sr = upscale_model(lr_image, model, args.device)
            metrics = compute_metrics(sr, hr_image, args.device)
            results[name]['psnr'].append(metrics['psnr'])
            results[name]['ssim'].append(metrics['ssim'])
            results[name]['lpips'].append(metrics['lpips'])
            sr_images[name] = sr
            all_metrics[name] = metrics

        # Save comparison images (every N images)
        if args.save_every > 0 and i % args.save_every == 0:
            lr_up = cv2.resize(lr_image, target_size, interpolation=cv2.INTER_NEAREST)

            def add_label(img, label, metrics):
                img_copy = img.copy()
                cv2.rectangle(img_copy, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(img_copy, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img_copy, f"PSNR: {metrics['psnr']:.2f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.putText(img_copy, f"SSIM: {metrics['ssim']:.4f}  LPIPS: {metrics['lpips']:.4f}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 255, 150), 1)
                return img_copy

            # Build panels: LR | OpenCV methods | Models | HR
            panels = [add_label(lr_up, "LR Input (64x64)", {'psnr': 0, 'ssim': 0, 'lpips': 1})]

            # Add OpenCV baselines
            for method_name in OPENCV_METHODS.keys():
                panels.append(add_label(sr_images[method_name], method_name, all_metrics[method_name]))

            # Add trained models
            for name in models.keys():
                panels.append(add_label(sr_images[name], name, all_metrics[name]))

            # HR ground truth
            hr_panel = hr_image.copy()
            cv2.rectangle(hr_panel, (0, 0), (hr_panel.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(hr_panel, "HR Ground Truth", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hr_panel, "256x256 original", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            panels.append(hr_panel)

            comparison = np.hstack(panels)
            cv2.imwrite(str(output_dir / f"comparison_{i:03d}.png"), comparison)

    # Print results
    print("\n" + "=" * 90)
    print("MODEL COMPARISON RESULTS")
    print("=" * 90)
    print(f"\n{'Method':<20} | {'Avg PSNR':>12} | {'Avg SSIM':>12} | {'Avg LPIPS':>12}")
    print("-" * 70)

    # Calculate averages
    averages = {}
    method_order = list(OPENCV_METHODS.keys()) + list(models.keys())

    # Print OpenCV baselines first
    print("--- Baseline Methods ---")
    for method in OPENCV_METHODS.keys():
        avg_psnr = np.mean(results[method]['psnr'])
        avg_ssim = np.mean(results[method]['ssim'])
        avg_lpips = np.mean(results[method]['lpips'])
        averages[method] = {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}
        print(f"{method:<20} | {avg_psnr:>10.2f} dB | {avg_ssim:>12.4f} | {avg_lpips:>12.4f}")

    # Print trained models
    print("-" * 70)
    print("--- Trained Models ---")
    for name in models.keys():
        avg_psnr = np.mean(results[name]['psnr'])
        avg_ssim = np.mean(results[name]['ssim'])
        avg_lpips = np.mean(results[name]['lpips'])
        averages[name] = {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}
        print(f"{name:<20} | {avg_psnr:>10.2f} dB | {avg_ssim:>12.4f} | {avg_lpips:>12.4f}")

    print("=" * 90)

    # Find best baseline for comparison
    best_baseline_name = max(OPENCV_METHODS.keys(), key=lambda x: averages[x]['psnr'])
    best_baseline = averages[best_baseline_name]

    print(f"\nBest baseline: {best_baseline_name} (PSNR: {best_baseline['psnr']:.2f} dB)")

    # Detailed comparison vs best baseline
    for name in models.keys():
        model_avg = averages[name]
        print(f"\n{name} vs {best_baseline_name}:")
        print(f"  PSNR:  {model_avg['psnr'] - best_baseline['psnr']:+.2f} dB")
        print(f"  SSIM:  {model_avg['ssim'] - best_baseline['ssim']:+.4f}")
        print(f"  LPIPS: {model_avg['lpips'] - best_baseline['lpips']:+.4f} (negative = better)")

    # If multiple models, compare them
    model_names = list(models.keys())
    if len(model_names) >= 2:
        print(f"\n{model_names[0]} vs {model_names[1]}:")
        m1, m2 = averages[model_names[0]], averages[model_names[1]]
        print(f"  PSNR:  {m1['psnr'] - m2['psnr']:+.2f} dB")
        print(f"  SSIM:  {m1['ssim'] - m2['ssim']:+.4f}")
        print(f"  LPIPS: {m1['lpips'] - m2['lpips']:+.4f}")

    print(f"\nComparison images saved to: {output_dir}/")

    # Save results to file for report
    results_file = output_dir / "results_summary.txt"
    with open(results_file, 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of test images: {len(image_files)}\n")
        f.write(f"Test set: {args.hr_dir}\n\n")
        f.write(f"{'Method':<20} | {'PSNR (dB)':>12} | {'SSIM':>12} | {'LPIPS':>12}\n")
        f.write("-" * 70 + "\n")

        f.write("Baseline Methods:\n")
        for method in OPENCV_METHODS.keys():
            avg = averages[method]
            f.write(f"{method:<20} | {avg['psnr']:>12.2f} | {avg['ssim']:>12.4f} | {avg['lpips']:>12.4f}\n")

        f.write("\nTrained Models:\n")
        for name in models.keys():
            avg = averages[name]
            f.write(f"{name:<20} | {avg['psnr']:>12.2f} | {avg['ssim']:>12.4f} | {avg['lpips']:>12.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"\nComparison vs {best_baseline_name} (best baseline):\n")
        for name in models.keys():
            model_avg = averages[name]
            f.write(f"\n{name}:\n")
            f.write(f"  PSNR:  {model_avg['psnr'] - best_baseline['psnr']:+.2f} dB\n")
            f.write(f"  SSIM:  {model_avg['ssim'] - best_baseline['ssim']:+.4f}\n")
            f.write(f"  LPIPS: {model_avg['lpips'] - best_baseline['lpips']:+.4f} (negative = better)\n")

    print(f"Results summary saved to: {results_file}")


if __name__ == '__main__':
    main()
