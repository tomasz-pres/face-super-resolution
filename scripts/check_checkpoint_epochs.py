#!/usr/bin/env python
"""
Check Training History in Checkpoints
======================================
Quick utility to see how many epochs are saved in each checkpoint.

Usage:
    python scripts/check_checkpoint_epochs.py
    python scripts/check_checkpoint_epochs.py checkpoints/my_model.pth
"""

import torch
import sys
from pathlib import Path

def check_checkpoint(checkpoint_path):
    """Check and display training history info from a checkpoint."""
    print(f"\n{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print('='*70)

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check for training history
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']

            # Count epochs
            if 'train_loss' in history:
                num_epochs = len(history['train_loss'])
                print(f"✓ Training history found: {num_epochs} epochs")

                # Show available metrics
                print(f"\nMetrics available:")
                for key in history.keys():
                    if isinstance(history[key], list):
                        non_none = sum(1 for x in history[key] if x is not None)
                        print(f"  - {key}: {non_none}/{num_epochs} non-null values")

                # Show epoch info if available
                if 'epoch' in checkpoint:
                    print(f"\nCheckpoint saved at epoch: {checkpoint['epoch']}")

                # Show first and last epoch metrics
                if num_epochs > 0:
                    print(f"\nFirst epoch (0):")
                    print(f"  Train Loss: {history['train_loss'][0]:.4f}")
                    if history['val_loss'][0] is not None:
                        print(f"  Val Loss:   {history['val_loss'][0]:.4f}")
                    if history['val_psnr'][0] is not None:
                        print(f"  Val PSNR:   {history['val_psnr'][0]:.2f} dB")

                    print(f"\nLast epoch ({num_epochs-1}):")
                    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
                    if history['val_loss'][-1] is not None:
                        print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
                    if history['val_psnr'][-1] is not None:
                        print(f"  Val PSNR:   {history['val_psnr'][-1]:.2f} dB")
            else:
                print("⚠ Training history exists but no train_loss found")
        else:
            print("✗ No training_history found in checkpoint")
            print(f"\nAvailable keys: {list(checkpoint.keys())}")

    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Check specific checkpoint
        checkpoint_path = sys.argv[1]
        check_checkpoint(checkpoint_path)
    else:
        # Check all checkpoints in checkpoints directory
        checkpoint_dir = Path('checkpoints')

        if not checkpoint_dir.exists():
            print(f"Error: {checkpoint_dir} not found")
            sys.exit(1)

        checkpoint_files = sorted(checkpoint_dir.glob('*.pth'))

        if not checkpoint_files:
            print(f"No .pth files found in {checkpoint_dir}")
            sys.exit(1)

        print(f"Found {len(checkpoint_files)} checkpoint(s):")
        for cp in checkpoint_files:
            print(f"  - {cp.name}")

        for checkpoint_path in checkpoint_files:
            check_checkpoint(str(checkpoint_path))

        print(f"\n{'='*70}")
