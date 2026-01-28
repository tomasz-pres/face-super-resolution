#!/usr/bin/env python
"""
Evaluation Script for Face Super-Resolution
=============================================
Evaluate trained models and generate comparison visualizations.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --compare-all
"""

import argparse
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm

from src.data import get_dataloader
from src.models import create_face_enhance_net, create_esrgan_baseline, FaceEnhanceNet
from src.evaluation import (
    MetricCalculator, create_comparison_grid, create_zoom_comparison,
    create_metrics_table, tensor_to_image
)


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Determine model type and create
    if 'config' in checkpoint:
        model = create_face_enhance_net(**checkpoint['config'])
    else:
        model = create_face_enhance_net()

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def bicubic_upscale(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Upscale using bicubic interpolation."""
    return torch.nn.functional.interpolate(
        lr, scale_factor=scale, mode='bicubic', align_corners=False
    )


def evaluate_model(model, dataloader, device: str = 'cuda',
                   desc: str = 'Evaluating') -> dict:
    """Evaluate model on dataset."""
    calculator = MetricCalculator(device=device)
    return calculator.evaluate_dataset(model, dataloader, desc=desc)


def generate_comparisons(models: dict, dataloader, output_dir: str,
                         num_samples: int = 5, device: str = 'cuda'):
    """Generate visual comparisons for multiple models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample batch
    batch = next(iter(dataloader))
    lr_batch = batch['lr'][:num_samples].to(device)
    hr_batch = batch['hr'][:num_samples].to(device)

    for idx in tqdm(range(num_samples), desc='Generating comparisons'):
        lr = lr_batch[idx:idx+1]
        hr = hr_batch[idx:idx+1]

        images = {
            'LR Input': tensor_to_image(
                torch.nn.functional.interpolate(lr, scale_factor=4, mode='nearest')
            ),
            'Bicubic': tensor_to_image(bicubic_upscale(lr)),
        }

        # Generate SR from each model
        for name, model in models.items():
            with torch.no_grad():
                sr = model(lr)
                sr = torch.clamp(sr, 0, 1)
            images[name] = tensor_to_image(sr)

        images['Ground Truth'] = tensor_to_image(hr)

        # Save comparison
        save_path = output_dir / f'comparison_{idx+1}.png'
        create_comparison_grid(images, save_path=str(save_path))

        # Create zoomed comparison
        h, w = hr.shape[2], hr.shape[3]
        zoom_region = (w//4, h//4, w//2, h//2)  # Center region
        zoom_path = output_dir / f'zoom_{idx+1}.png'
        create_zoom_comparison(images, zoom_region, save_path=str(zoom_path))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Super-Resolution Models')

    parser.add_argument('--checkpoint', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='data/processed',
                        help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples for visual comparison')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all available models')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"\nLoading test data from {args.data_root}...")
    test_loader = get_dataloader(
        args.data_root,
        mode='test',
        batch_size=args.batch_size,
        num_workers=2
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate models
    results = {}
    models = {}

    if args.checkpoint:
        # Single model evaluation
        print(f"\nLoading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, device)
        models['Custom Model'] = model

        print("\nEvaluating model...")
        results['Custom Model'] = evaluate_model(model, test_loader, device)

    if args.compare_all:
        # Load all available models
        print("\nLoading baseline model...")
        try:
            baseline = create_esrgan_baseline(device=device)
            models['ESRGAN Baseline'] = baseline
            print("Evaluating ESRGAN baseline...")
            results['ESRGAN Baseline'] = evaluate_model(baseline, test_loader, device)
        except Exception as e:
            print(f"Could not load baseline: {e}")

        # Check for other checkpoints
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob('*.pth'):
                if 'best' in ckpt.name.lower():
                    print(f"\nLoading {ckpt.name}...")
                    try:
                        model = load_model(str(ckpt), device)
                        name = ckpt.stem
                        models[name] = model
                        results[name] = evaluate_model(model, test_loader, device)
                    except Exception as e:
                        print(f"Could not load {ckpt}: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        if 'lpips_mean' in metrics:
            print(f"  LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")

    # Save results
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Create metrics table
    table = create_metrics_table(
        {name: {'PSNR': m['psnr_mean'], 'SSIM': m['ssim_mean']}
         for name, m in results.items()},
        save_path=str(output_dir / 'metrics_table.md')
    )
    print("\n" + table)

    # Generate visual comparisons
    if models:
        print("\nGenerating visual comparisons...")
        generate_comparisons(
            models, test_loader, str(output_dir / 'comparisons'),
            num_samples=args.num_samples, device=device
        )

    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
