#!/usr/bin/env python
"""
Generate Training Curves for Report
====================================
Creates publication-ready plots from training history.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --input training_history.csv
    python scripts/plot_training_curves.py --input training_history.csv --output reports/figures
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def plot_training_curves(input_file: str, output_dir: str):
    """Generate all training curve plots."""

    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return

    print(f"Reading training data from: {input_file}")
    df = pd.read_csv(input_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_path}")

    # Determine training stages based on epochs
    epochs = np.arange(len(df))
    stage1_end = 99  # PSNR pre-training
    stage2_end = 149  # SSIM fine-tuning
    # Stage 3: GAN (150+)

    # Figure 1: Training and Validation Loss
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, df['train_loss'], label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, df['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)

    # Mark training stages
    ax.axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1 → 2')
    ax.axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5, label='Stage 2 → 3 (GAN)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_loss.png'}")
    plt.close()

    # Figure 2: Validation PSNR
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, df['val_psnr'], linewidth=2, color='green', alpha=0.8)

    # Mark training stages
    ax.axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1 → 2')
    ax.axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5, label='Stage 2 → 3 (GAN)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add max PSNR annotation
    max_psnr = df['val_psnr'].max()
    max_epoch = df['val_psnr'].idxmax()
    ax.annotate(f'Max: {max_psnr:.2f} dB\n(Epoch {max_epoch})',
                xy=(max_epoch, max_psnr),
                xytext=(max_epoch + 10, max_psnr - 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_psnr.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'validation_psnr.png'}")
    plt.close()

    # Figure 3: Validation SSIM
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, df['val_ssim'], linewidth=2, color='blue', alpha=0.8)

    # Mark training stages
    ax.axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1 → 2')
    ax.axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5, label='Stage 2 → 3 (GAN)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add max SSIM annotation
    max_ssim = df['val_ssim'].max()
    max_epoch_ssim = df['val_ssim'].idxmax()
    ax.annotate(f'Max: {max_ssim:.4f}\n(Epoch {max_epoch_ssim})',
                xy=(max_epoch_ssim, max_ssim),
                xytext=(max_epoch_ssim + 10, max_ssim - 0.005),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_ssim.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'validation_ssim.png'}")
    plt.close()

    # Figure 4: Learning Rate Schedule
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, df['learning_rate'], linewidth=2, color='purple', alpha=0.8)

    # Mark training stages
    ax.axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1 → 2')
    ax.axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.5, label='Stage 2 → 3 (GAN)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Annealing)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visibility

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'learning_rate.png'}")
    plt.close()

    # Figure 5: Combined Metrics (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, df['val_loss'], label='Val', linewidth=2)
    axes[0, 0].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PSNR
    axes[0, 1].plot(epochs, df['val_psnr'], linewidth=2, color='green')
    axes[0, 1].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('Validation PSNR')
    axes[0, 1].grid(True, alpha=0.3)

    # SSIM
    axes[1, 0].plot(epochs, df['val_ssim'], linewidth=2, color='blue')
    axes[1, 0].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('Validation SSIM')
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(epochs, df['learning_rate'], linewidth=2, color='purple')
    axes[1, 1].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.3, label='Stage 1→2')
    axes[1, 1].axvline(x=stage2_end, color='orange', linestyle='--', alpha=0.3, label='Stage 2→3')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_summary.png'}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total Epochs: {len(df)}")
    print(f"\nFinal Metrics (Epoch {len(df)-1}):")
    print(f"  Training Loss:   {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"  Validation PSNR: {df['val_psnr'].iloc[-1]:.2f} dB")
    print(f"  Validation SSIM: {df['val_ssim'].iloc[-1]:.4f}")
    print(f"\nBest Metrics:")
    print(f"  Best Val Loss:   {df['val_loss'].min():.4f} (Epoch {df['val_loss'].idxmin()})")
    print(f"  Best Val PSNR:   {df['val_psnr'].max():.2f} dB (Epoch {df['val_psnr'].idxmax()})")
    print(f"  Best Val SSIM:   {df['val_ssim'].max():.4f} (Epoch {df['val_ssim'].idxmax()})")
    print("=" * 60)

if __name__ == '__main__':
    plot_training_curves()
    print("\nAll plots generated successfully!")
