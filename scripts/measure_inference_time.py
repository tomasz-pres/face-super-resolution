"""
Measure inference time for the super-resolution models.
"""

import torch
import time
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FaceEnhanceNet, create_transfer_model


def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def get_model_config_from_checkpoint(checkpoint_path):
    """Detect model architecture from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt

    # Detect model type from keys
    first_key = list(state.keys())[0]

    if first_key.startswith('backbone.'):
        # TransferSRModel
        return {'type': 'transfer'}

    # FaceEnhanceNet - find groups and blocks from keys
    max_group = 0
    max_block = 0
    for k in state.keys():
        if 'residual_groups' in k:
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'residual_groups':
                    max_group = max(max_group, int(parts[i+1]))
                if p == 'blocks':
                    max_block = max(max_block, int(parts[i+1]))

    # Get channels from conv_first
    channels = state['conv_first.weight'].shape[0]

    return {
        'type': 'custom',
        'num_channels': channels,
        'num_groups': max_group + 1,
        'blocks_per_group': max_block + 1,
        'scale_factor': 4
    }


def measure_inference_time(model, input_size=(1, 3, 64, 64), device='cuda', num_runs=100, warmup=10):
    """
    Measure average inference time.

    Args:
        model: The model to measure
        input_size: Input tensor size (B, C, H, W)
        device: 'cuda' or 'cpu'
        num_runs: Number of runs to average
        warmup: Number of warmup runs (not counted)

    Returns:
        Average inference time in milliseconds
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize before timing (for GPU)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, min_time, max_time


def main():
    parser = argparse.ArgumentParser(description='Measure model inference time')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-runs', type=int, default=100)
    parser.add_argument('--custom-checkpoint', type=str, default=None,
                        help='Path to custom model checkpoint')
    parser.add_argument('--transfer-checkpoint', type=str, default=None,
                        help='Path to transfer model checkpoint')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Number of runs: {args.num_runs}")
    print("-" * 50)

    # Test Custom model (GAN fine-tuned)
    if args.custom_checkpoint:
        print(f"\n1. Custom Model (from {args.custom_checkpoint})")
        config = get_model_config_from_checkpoint(args.custom_checkpoint)

        if config.get('type') == 'transfer':
            print(f"   Architecture: TransferSRModel (ESRGAN backbone)")
            custom_model = create_transfer_model()
        else:
            print(f"   Architecture: {config['num_groups']} groups x {config['blocks_per_group']} blocks")
            custom_model = FaceEnhanceNet(**{k: v for k, v in config.items() if k != 'type'})

        custom_model = load_checkpoint(custom_model, args.custom_checkpoint)
    else:
        print("\n1. FaceEnhanceNet (6 groups x 10 blocks)")
        custom_model = FaceEnhanceNet(
            num_channels=64,
            num_groups=6,
            blocks_per_group=10,
            scale_factor=4
        )

    num_params = sum(p.numel() for p in custom_model.parameters())
    size_mb = num_params * 4 / (1024 ** 2)
    print(f"   Parameters: {num_params:,}")
    print(f"   Size: {size_mb:.2f} MB")

    avg, min_t, max_t = measure_inference_time(
        custom_model, device=device, num_runs=args.num_runs
    )
    print(f"   Inference time: {avg:.2f} ms (min: {min_t:.2f}, max: {max_t:.2f})")

    # Test Transfer model
    if args.transfer_checkpoint:
        print(f"\n2. Transfer Model (from {args.transfer_checkpoint})")
        config = get_model_config_from_checkpoint(args.transfer_checkpoint)

        if config.get('type') == 'transfer':
            print(f"   Architecture: TransferSRModel (ESRGAN backbone)")
            transfer_model = create_transfer_model()
        else:
            print(f"   Architecture: {config['num_groups']} groups x {config['blocks_per_group']} blocks")
            transfer_model = FaceEnhanceNet(**{k: v for k, v in config.items() if k != 'type'})

        transfer_model = load_checkpoint(transfer_model, args.transfer_checkpoint)

        num_params = sum(p.numel() for p in transfer_model.parameters())
        size_mb = num_params * 4 / (1024 ** 2)
        print(f"   Parameters: {num_params:,}")
        print(f"   Size: {size_mb:.2f} MB")

        avg, min_t, max_t = measure_inference_time(
            transfer_model, device=device, num_runs=args.num_runs
        )
        print(f"   Inference time: {avg:.2f} ms (min: {min_t:.2f}, max: {max_t:.2f})")

    print("\n" + "-" * 50)
    print("Done!")


if __name__ == '__main__':
    main()
