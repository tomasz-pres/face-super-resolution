#!/usr/bin/env python
"""
Generate Training Stage Plots from Checkpoint
==============================================
Creates separate plots for each training stage.

Usage:
    python scripts/plot_training_stages.py
    python scripts/plot_training_stages.py --checkpoint checkpoints/final_custom_model.pth
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11

def plot_stage(epochs, history, stage_name, stage_epochs, output_dir, has_gan=False):
    """Plot metrics for a specific training stage."""

    start_epoch, end_epoch = stage_epochs
    stage_slice = slice(start_epoch, end_epoch + 1)
    stage_epochs_arr = epochs[stage_slice]

    # Figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{stage_name}', fontsize=16, fontweight='bold')

    # Plot 1: Train & Val Loss
    ax = axes[0, 0]
    ax.plot(stage_epochs_arr, history['train_loss'][stage_slice],
            label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(stage_epochs_arr, history['val_loss'][stage_slice],
            label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation PSNR
    ax = axes[0, 1]
    ax.plot(stage_epochs_arr, history['val_psnr'][stage_slice],
            linewidth=2, color='green', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR')
    ax.grid(True, alpha=0.3)

    # Add best PSNR annotation
    stage_psnr = np.array(history['val_psnr'][stage_slice])
    if len(stage_psnr) > 0:
        max_psnr_idx = np.argmax(stage_psnr)
        max_psnr = stage_psnr[max_psnr_idx]
        max_epoch = stage_epochs_arr[max_psnr_idx]
        ax.plot(max_epoch, max_psnr, 'r*', markersize=15, label=f'Best: {max_psnr:.2f} dB')
        ax.legend()

    # Plot 3: Validation SSIM
    ax = axes[1, 0]
    ax.plot(stage_epochs_arr, history['val_ssim'][stage_slice],
            linewidth=2, color='blue', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM')
    ax.grid(True, alpha=0.3)

    # Add best SSIM annotation
    stage_ssim = np.array(history['val_ssim'][stage_slice])
    if len(stage_ssim) > 0:
        max_ssim_idx = np.argmax(stage_ssim)
        max_ssim = stage_ssim[max_ssim_idx]
        max_epoch_ssim = stage_epochs_arr[max_ssim_idx]
        ax.plot(max_epoch_ssim, max_ssim, 'r*', markersize=15, label=f'Best: {max_ssim:.4f}')
        ax.legend()

    # Plot 4: Learning Rate or GAN Loss
    ax = axes[1, 1]
    if has_gan and 'd_loss' in history and 'g_loss' in history:
        # Plot GAN losses
        d_loss_stage = history['d_loss'][stage_slice]
        g_loss_stage = history['g_loss'][stage_slice]

        # Convert to list if needed and filter out None values
        d_loss_list = list(d_loss_stage) if not isinstance(d_loss_stage, list) else d_loss_stage
        g_loss_list = list(g_loss_stage) if not isinstance(g_loss_stage, list) else g_loss_stage

        # Filter out None values
        d_valid_indices = [i for i, val in enumerate(d_loss_list) if val is not None and not np.isnan(val)]
        g_valid_indices = [i for i, val in enumerate(g_loss_list) if val is not None and not np.isnan(val)]

        print(f"    Debug: Found {len(d_valid_indices)} valid D_loss values, {len(g_valid_indices)} valid G_loss values")

        if len(d_valid_indices) > 0:
            d_epochs = stage_epochs_arr[d_valid_indices]
            d_values = [d_loss_list[i] for i in d_valid_indices]
            ax.plot(d_epochs, d_values,
                    label='Discriminator Loss', linewidth=2, marker='o', markersize=3, color='red')

        if len(g_valid_indices) > 0:
            g_epochs = stage_epochs_arr[g_valid_indices]
            g_values = [g_loss_list[i] for i in g_valid_indices]
            ax.plot(g_epochs, g_values,
                    label='Generator Loss', linewidth=2, marker='s', markersize=3, color='purple')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('GAN Losses')
        if len(d_valid_indices) > 0 or len(g_valid_indices) > 0:
            ax.legend()
        else:
            print("    Warning: No valid GAN loss values found, plotting learning rate instead")
            ax.plot(stage_epochs_arr, history['learning_rate'][stage_slice],
                    linewidth=2, color='purple', marker='o', markersize=3)
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
    else:
        # Plot learning rate
        ax.plot(stage_epochs_arr, history['learning_rate'][stage_slice],
                linewidth=2, color='purple', marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f"{stage_name.lower().replace(' ', '_').replace(':', '')}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def combine_histories(hist1, hist2, offset):
    """Combine two training histories, with hist2 offset by offset epochs."""
    combined = {}

    for key in hist1.keys():
        combined[key] = hist1[key].copy()

        # Add hist2 data with offset
        if key in hist2:
            # For hist2, we need to offset the epochs
            combined[key].extend(hist2[key])

    return combined


def plot_training_stages(checkpoint_stage1: str, checkpoint_stage2: str, checkpoint_stage3: str, output_dir: str):
    """Generate plots for each training stage from 3 separate checkpoints."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_path}\n")

    histories = {}
    epoch_offsets = {}

    # Load Stage 1 checkpoint (PSNR pre-training, epochs 0-99)
    print(f"Loading Stage 1 checkpoint: {checkpoint_stage1}")
    cp1 = torch.load(checkpoint_stage1, map_location='cpu')
    if 'training_history' in cp1:
        histories['stage1'] = cp1['training_history']
        num_epochs_1 = len(histories['stage1']['train_loss'])
        epoch_offsets['stage1'] = 0
        print(f"  ✓ Found {num_epochs_1} epochs for Stage 1")
    else:
        print(f"  ✗ No training_history found")

    # Load Stage 2 checkpoint (SSIM fine-tuning, epochs 100-149)
    print(f"\nLoading Stage 2 checkpoint: {checkpoint_stage2}")
    cp2 = torch.load(checkpoint_stage2, map_location='cpu')
    if 'training_history' in cp2:
        histories['stage2'] = cp2['training_history']
        num_epochs_2 = len(histories['stage2']['train_loss'])
        epoch_offsets['stage2'] = 100
        print(f"  ✓ Found {num_epochs_2} epochs for Stage 2")
    else:
        print(f"  ✗ No training_history found")

    # Load Stage 3 checkpoint (GAN training, epochs 150-169)
    print(f"\nLoading Stage 3 checkpoint: {checkpoint_stage3}")
    cp3 = torch.load(checkpoint_stage3, map_location='cpu')
    if 'training_history' in cp3:
        histories['stage3'] = cp3['training_history']
        num_epochs_3 = len(histories['stage3']['train_loss'])
        epoch_offsets['stage3'] = 150
        print(f"  ✓ Found {num_epochs_3} epochs for Stage 3")
    else:
        print(f"  ✗ No training_history found")

    print()

    # Plot Stage 1: PSNR Pre-training (Epochs 0-99)
    if 'stage1' in histories:
        print("Plotting Stage 1: PSNR Pre-training (Epochs 0-99)...")
        history = histories['stage1']
        num_epochs = len(history['train_loss'])
        epochs = np.arange(num_epochs)
        stage_range = (0, num_epochs - 1)
        plot_stage(epochs, history, "Stage 1: PSNR Pre-training (Epochs 0-99)",
                   stage_range, output_path, has_gan=False)

    # Plot Stage 2: SSIM Fine-tuning (Epochs 100-149)
    if 'stage2' in histories:
        print("Plotting Stage 2: SSIM Fine-tuning (Epochs 100-149)...")
        history = histories['stage2']
        num_epochs = len(history['train_loss'])
        epochs = np.arange(100, 100 + num_epochs)  # Offset to start at epoch 100
        stage_range = (0, num_epochs - 1)
        plot_stage(epochs, history, "Stage 2: SSIM Fine-tuning (Epochs 100-149)",
                   stage_range, output_path, has_gan=False)

    # Plot Stage 3: GAN Fine-tuning (Epochs 150-169)
    if 'stage3' in histories:
        print("Plotting Stage 3: GAN Fine-tuning (Epochs 150-169)...")
        history = histories['stage3']
        num_epochs = len(history['train_loss'])
        epochs = np.arange(150, 150 + num_epochs)  # Offset to start at epoch 150
        stage_range = (0, num_epochs - 1)
        plot_stage(epochs, history, "Stage 3: GAN Fine-tuning (Epochs 150-169)",
                   stage_range, output_path, has_gan=True)

    # Combine all histories for overview
    combined_history = {}
    if 'stage1' in histories:
        for key in histories['stage1'].keys():
            combined_history[key] = histories['stage1'][key].copy()

    if 'stage2' in histories:
        for key in histories['stage2'].keys():
            if key in combined_history:
                combined_history[key].extend(histories['stage2'][key])
            else:
                combined_history[key] = histories['stage2'][key].copy()

    if 'stage3' in histories:
        for key in histories['stage3'].keys():
            if key in combined_history:
                combined_history[key].extend(histories['stage3'][key])
            else:
                combined_history[key] = histories['stage3'][key].copy()

    total_epochs = len(combined_history.get('train_loss', []))

    # Combined overview plot
    if total_epochs > 0:
        print("\nCreating combined overview...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Overview - All Stages', fontsize=18, fontweight='bold')

        epochs_combined = np.arange(total_epochs)

        # Mark stage boundaries
        stage1_end = 99
        stage2_end = 149

        # Loss
        axes[0, 0].plot(epochs_combined, combined_history['train_loss'], label='Train', linewidth=2, alpha=0.8)
        axes[0, 0].plot(epochs_combined, combined_history['val_loss'], label='Val', linewidth=2, alpha=0.8)
        axes[0, 0].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
        axes[0, 0].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5, label='Stage 2→3')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # PSNR
        axes[0, 1].plot(epochs_combined, combined_history['val_psnr'], linewidth=2, color='green', alpha=0.8)
        axes[0, 1].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Validation PSNR')
        axes[0, 1].grid(True, alpha=0.3)

        # SSIM
        axes[1, 0].plot(epochs_combined, combined_history['val_ssim'], linewidth=2, color='blue', alpha=0.8)
        axes[1, 0].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Validation SSIM')
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs_combined, combined_history['learning_rate'], linewidth=2, color='purple', alpha=0.8)
        axes[1, 1].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
        axes[1, 1].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5, label='Stage 2→3')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        overview_path = output_path / 'training_overview.png'
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {overview_path}")
        plt.close()

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Epochs: {total_epochs}")
    if total_epochs > 0:
        print(f"\nFinal Metrics:")
        print(f"  Val Loss: {combined_history['val_loss'][-1]:.4f}")
        print(f"  Val PSNR: {combined_history['val_psnr'][-1]:.2f} dB")
        print(f"  Val SSIM: {combined_history['val_ssim'][-1]:.4f}")
        print(f"\nBest Metrics:")
        print(f"  Best PSNR: {max(combined_history['val_psnr']):.2f} dB (Epoch {np.argmax(combined_history['val_psnr'])})")
        print(f"  Best SSIM: {max(combined_history['val_ssim']):.4f} (Epoch {np.argmax(combined_history['val_ssim'])})")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training stages from 3 separate checkpoints')
    parser.add_argument('--stage1', type=str, default='checkpoints/big_model_after_epoch_100.pth',
                        help='Checkpoint with Stage 1 (PSNR pre-training, epochs 0-99)')
    parser.add_argument('--stage2', type=str, default='checkpoints/epoch_50.pth',
                        help='Checkpoint with Stage 2 (SSIM fine-tuning, epochs 100-149)')
    parser.add_argument('--stage3', type=str, default='checkpoints/final_custom_model.pth',
                        help='Checkpoint with Stage 3 (GAN training, epochs 150-169)')
    parser.add_argument('--output', type=str, default='reports/figures',
                        help='Output directory for plots')

    args = parser.parse_args()

    plot_training_stages(args.stage1, args.stage2, args.stage3, args.output)

    print("\n✓ All plots generated successfully!")
