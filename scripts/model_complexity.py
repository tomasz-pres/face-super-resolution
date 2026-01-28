#!/usr/bin/env python
"""
Model Complexity Analysis
=========================
Calculate parameter count, model size, and inference time.

Usage:
    python scripts/model_complexity.py
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_face_enhance_net


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(checkpoint_path):
    """Get model file size in MB."""
    size_bytes = Path(checkpoint_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


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


def benchmark_inference(model, device='cuda', num_runs=100):
    """Benchmark inference time."""
    model.eval()

    # Warm up
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.time()
                _ = model(dummy_input)
                end = time.time()
                times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def analyze_model(checkpoint_path, model_name, device='cuda'):
    """Analyze a single model."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print('='*70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Infer config
    config = infer_model_config_from_state_dict(state_dict)
    print(f"\nModel Config:")
    print(f"  Residual Groups: {config['num_groups']}")
    print(f"  RCAB Blocks per Group: {config['blocks_per_group']}")
    print(f"  Feature Channels: {config['num_channels']}")

    # Create model
    model = create_face_enhance_net(**config)
    model.load_state_dict(state_dict)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Model size
    file_size = get_model_size_mb(checkpoint_path)
    print(f"\nModel File Size: {file_size:.2f} MB")

    # Inference time
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        print(f"\nBenchmarking on GPU...")
        mean_time, std_time = benchmark_inference(model, device='cuda')
        print(f"  GPU Inference Time: {mean_time:.2f} ± {std_time:.2f} ms/image")

    model = model.to('cpu')
    print(f"\nBenchmarking on CPU...")
    mean_time_cpu, std_time_cpu = benchmark_inference(model, device='cpu', num_runs=50)
    print(f"  CPU Inference Time: {mean_time_cpu:.2f} ± {std_time_cpu:.2f} ms/image")

    return {
        'name': model_name,
        'params': total_params,
        'size_mb': file_size,
        'gpu_time': mean_time if device == 'cuda' and torch.cuda.is_available() else None,
        'cpu_time': mean_time_cpu,
        'config': config
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    models = [
        ('checkpoints/final_custom_model.pth', 'Custom (GAN)'),
        ('checkpoints/transfer_best.pth', 'Transfer Learning'),
    ]

    results = []
    for checkpoint_path, model_name in models:
        if Path(checkpoint_path).exists():
            result = analyze_model(checkpoint_path, model_name, device)
            results.append(result)
        else:
            print(f"\n⚠ Skipping {model_name}: checkpoint not found")

    # Print summary table
    print("\n" + "="*90)
    print("MODEL COMPLEXITY COMPARISON")
    print("="*90)
    print(f"\n{'Model':<20} | {'Parameters':>12} | {'Size (MB)':>10} | {'GPU (ms)':>10} | {'CPU (ms)':>10}")
    print("-"*90)

    # Add baselines
    print(f"{'Bicubic':<20} | {'0':>12} | {'0':>10} | {'~1':>10} | {'~1':>10}")
    print(f"{'Bilinear':<20} | {'0':>12} | {'0':>10} | {'<1':>10} | {'<1':>10}")
    print(f"{'Lanczos4':<20} | {'0':>12} | {'0':>10} | {'~2':>10} | {'~2':>10}")
    print("-"*90)

    for result in results:
        params_m = f"{result['params']/1e6:.2f}M"
        gpu_time = f"{result['gpu_time']:.2f}" if result['gpu_time'] else "N/A"
        cpu_time = f"{result['cpu_time']:.2f}"
        print(f"{result['name']:<20} | {params_m:>12} | {result['size_mb']:>10.2f} | {gpu_time:>10} | {cpu_time:>10}")

    print("="*90)

    # Save to file
    output_file = Path('reports/model_complexity.txt')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("MODEL COMPLEXITY COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Method':<20} | {'Parameters':>12} | {'Size (MB)':>10} | {'Inference Time':>15}\n")
        f.write("-"*70 + "\n")
        f.write("Baseline Methods:\n")
        f.write(f"{'Bicubic':<20} | {'0':>12} | {'0':>10} | {'~1 ms':>15}\n")
        f.write(f"{'Bilinear':<20} | {'0':>12} | {'0':>10} | {'<1 ms':>15}\n")
        f.write(f"{'Lanczos4':<20} | {'0':>12} | {'0':>10} | {'~2 ms':>15}\n")
        f.write("\nDeep Learning Models:\n")
        for result in results:
            params_m = f"{result['params']/1e6:.2f}M"
            time_str = f"{result['gpu_time']:.2f} ms (GPU)" if result['gpu_time'] else f"{result['cpu_time']:.2f} ms (CPU)"
            f.write(f"{result['name']:<20} | {params_m:>12} | {result['size_mb']:>10.2f} | {time_str:>15}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
