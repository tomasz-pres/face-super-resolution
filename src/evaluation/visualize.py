"""
Visualization Utilities for Super-Resolution
==============================================
Comparison grids, zoomed regions, and publication-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import cv2


def create_comparison_grid(images: Dict[str, np.ndarray],
                           metrics: Optional[Dict[str, Dict[str, float]]] = None,
                           figsize: Tuple[int, int] = (15, 5),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create side-by-side comparison grid.

    Args:
        images: Dictionary of {name: image} pairs
        metrics: Optional metrics to display under each image
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_images = len(images)

    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.axis('off')

        title = name
        if metrics and name in metrics:
            m = metrics[name]
            title += f"\nPSNR: {m.get('psnr', 0):.2f} dB"
            if 'ssim' in m:
                title += f", SSIM: {m['ssim']:.3f}"

        ax.set_title(title, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    return fig


def create_zoom_comparison(images: Dict[str, np.ndarray],
                           region: Tuple[int, int, int, int],
                           zoom_factor: float = 2.0,
                           figsize: Tuple[int, int] = (15, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison with zoomed regions.

    Args:
        images: Dictionary of {name: image} pairs
        region: (x, y, width, height) of zoom region
        zoom_factor: Magnification factor
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    x, y, w, h = region

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, n_images, height_ratios=[2, 1])

    for idx, (name, img) in enumerate(images.items()):
        # Full image with rectangle
        ax_full = fig.add_subplot(gs[0, idx])
        ax_full.imshow(img)
        ax_full.add_patch(plt.Rectangle((x, y), w, h,
                                        fill=False, edgecolor='red', linewidth=2))
        ax_full.axis('off')
        ax_full.set_title(name, fontsize=12)

        # Zoomed region
        ax_zoom = fig.add_subplot(gs[1, idx])
        zoomed = img[y:y+h, x:x+w]
        ax_zoom.imshow(zoomed)
        ax_zoom.axis('off')
        ax_zoom.set_title('Zoomed Region', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved zoom comparison to {save_path}")

    return fig


def create_metrics_table(results: Dict[str, Dict[str, float]],
                         save_path: Optional[str] = None) -> str:
    """
    Create formatted metrics table.

    Args:
        results: {model_name: {metric: value}}
        save_path: Path to save as text file

    Returns:
        Formatted table string
    """
    # Get all metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    metrics_list = sorted(all_metrics)

    # Create header
    header = "| Model | " + " | ".join(metrics_list) + " |"
    separator = "|" + "---|" * (len(metrics_list) + 1)

    # Create rows
    rows = []
    for model, metrics in results.items():
        row = f"| {model} |"
        for metric in metrics_list:
            val = metrics.get(metric, '-')
            if isinstance(val, float):
                if 'psnr' in metric.lower():
                    row += f" {val:.2f} dB |"
                else:
                    row += f" {val:.4f} |"
            else:
                row += f" {val} |"
        rows.append(row)

    table = "\n".join([header, separator] + rows)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
        print(f"Saved metrics table to {save_path}")

    return table


def plot_training_curves(history: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (12, 4),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training curves from history.

    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # PSNR
    if 'val_psnr' in history:
        axes[1].plot(history['val_psnr'], color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Validation PSNR')
        axes[1].grid(True, alpha=0.3)

    # SSIM
    if 'val_ssim' in history:
        axes[2].plot(history['val_ssim'], color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('SSIM')
        axes[2].set_title('Validation SSIM')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def tensor_to_image(tensor: 'torch.Tensor') -> np.ndarray:
    """
    Convert tensor to numpy image.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)

    Returns:
        Numpy array (H, W, C) uint8
    """
    import torch

    if tensor.dim() == 4:
        tensor = tensor[0]

    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # CHW to HWC
    image = tensor.permute(1, 2, 0).cpu().numpy()

    # Convert to uint8
    image = (image * 255).astype(np.uint8)

    return image


def save_sr_result(lr: np.ndarray, sr: np.ndarray, hr: np.ndarray,
                   save_path: str, metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Save super-resolution result as comparison image.

    Args:
        lr: Low resolution input
        sr: Super-resolved output
        hr: High resolution ground truth
        save_path: Path to save
        metrics: Optional metrics to display
    """
    images = {
        'LR (Input)': cv2.resize(lr, (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST),
        'SR (Output)': sr,
        'HR (Target)': hr
    }

    metric_dict = None
    if metrics:
        metric_dict = {'SR (Output)': metrics}

    fig = create_comparison_grid(images, metric_dict, save_path=save_path)
    plt.close(fig)
