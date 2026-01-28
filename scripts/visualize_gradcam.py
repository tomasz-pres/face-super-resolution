#!/usr/bin/env python
"""
GradCAM Visualization Script
=============================
Generate attention heatmaps showing what the model focuses on.

Usage:
    python scripts/visualize_gradcam.py --checkpoint checkpoints/best_model.pth --image data/processed/val/HR/00001.png
    python scripts/visualize_gradcam.py --checkpoint checkpoints/best_model.pth --num-images 5
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
from tqdm import tqdm

from src.models import create_face_enhance_net
from src.explainability import (
    GradCAM,
    apply_heatmap,
    create_gradcam_visualization,
    visualize_attention_flow,
)


def infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model configuration from checkpoint weights."""
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
    for key in state_dict.keys():
        if key.startswith('residual_groups.') and '.blocks.' in key:
            group_idx = int(key.split('.')[1])
            group_indices.add(group_idx)
    if group_indices:
        config['num_groups'] = max(group_indices) + 1

    # Count blocks per group
    block_indices = set()
    for key in state_dict.keys():
        if 'residual_groups.0.blocks.' in key:
            parts = key.split('.')
            block_idx_pos = parts.index('blocks') + 1
            if block_idx_pos < len(parts):
                try:
                    block_idx = int(parts[block_idx_pos])
                    block_indices.add(block_idx)
                except ValueError:
                    pass
    if block_indices:
        config['blocks_per_group'] = max(block_indices) + 1

    # Get num_channels from conv_first
    if 'conv_first.weight' in state_dict:
        config['num_channels'] = state_dict['conv_first.weight'].shape[0]

    # Get reduction_ratio from channel attention
    ca_key = 'residual_groups.0.blocks.0.channel_attention.fc.0.weight'
    if ca_key in state_dict:
        reduced_channels = state_dict[ca_key].shape[0]
        config['reduction_ratio'] = config['num_channels'] // reduced_channels

    return config


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint with auto-detected config."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_config = infer_model_config_from_state_dict(state_dict)
    print(f"Model config: {model_config}")

    model = create_face_enhance_net(**model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def generate_lr(hr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """Generate LR image using F.interpolate (bicubic) to match training."""
    hr_tensor = torch.from_numpy(hr_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_tensor = F.interpolate(hr_tensor, size=(lr_h, lr_w), mode='bicubic', align_corners=False)

    lr_np = lr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    lr_np = np.clip(lr_np * 255, 0, 255).astype(np.uint8)

    return lr_np


def create_region_comparison(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    hr_image: np.ndarray,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Create visualization comparing attention for different face regions.
    """
    # Prepare input
    lr_tensor = torch.from_numpy(lr_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_tensor = lr_tensor.to(device)

    # Get SR output
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_image = sr_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    sr_image = np.clip(sr_image * 255, 0, 255).astype(np.uint8)

    # Initialize GradCAM
    gradcam = GradCAM(model, ['residual_groups'])

    regions = {
        'Full Image': 'full',
        'Center': 'center',
        'Eyes Region': 'eyes',
        'Mouth Region': 'mouth',
    }

    panels = []

    # LR Input
    lr_display = cv2.resize(lr_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    lr_panel = np.ascontiguousarray(lr_display.copy())
    cv2.rectangle(lr_panel, (0, 0), (256, 30), (0, 0, 0), -1)
    cv2.putText(lr_panel, "LR Input (64x64)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(lr_panel)

    # Attention for each region
    for label, region in regions.items():
        try:
            heatmap, _ = gradcam.generate_cam(lr_tensor, target_region=region)
            heatmap_up = cv2.resize(heatmap, (256, 256))
            overlay = apply_heatmap(lr_display.copy(), heatmap_up, alpha=0.5)
            overlay = np.ascontiguousarray(overlay)

            cv2.rectangle(overlay, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(overlay, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels.append(overlay)
        except Exception as e:
            print(f"Warning: Could not generate CAM for {label}: {e}")

    # SR Output
    sr_panel = np.ascontiguousarray(sr_image.copy())
    cv2.rectangle(sr_panel, (0, 0), (256, 30), (0, 0, 0), -1)
    cv2.putText(sr_panel, "SR Output (256x256)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(sr_panel)

    # HR Ground Truth
    if hr_image is not None:
        hr_panel = np.ascontiguousarray(hr_image.copy())
        cv2.rectangle(hr_panel, (0, 0), (256, 30), (0, 0, 0), -1)
        cv2.putText(hr_panel, "HR Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(hr_panel)

    gradcam.remove_hooks()

    # Stack panels
    visualization = np.hstack(panels)

    return visualization


def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    output_dir: Path,
    device: str = 'cuda'
):
    """Process a single image and save GradCAM visualizations."""
    # Load HR image
    hr_image = cv2.imread(image_path)
    if hr_image is None:
        print(f"Could not load {image_path}")
        return

    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

    # Resize to 256x256 if needed
    if hr_image.shape[0] != 256 or hr_image.shape[1] != 256:
        hr_image = cv2.resize(hr_image, (256, 256), interpolation=cv2.INTER_AREA)

    # Generate LR
    lr_image = generate_lr(hr_image, scale_factor=4)

    # Create visualizations
    image_name = Path(image_path).stem

    # 1. Region comparison
    region_viz = create_region_comparison(model, lr_image, hr_image, device)
    region_path = output_dir / f"{image_name}_gradcam_regions.png"
    cv2.imwrite(str(region_path), cv2.cvtColor(region_viz, cv2.COLOR_RGB2BGR))

    # 2. Attention flow through layers
    flow_viz = visualize_attention_flow(model, lr_image, device)
    flow_path = output_dir / f"{image_name}_gradcam_flow.png"
    cv2.imwrite(str(flow_path), cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR))

    print(f"Saved: {region_path.name}, {flow_path.name}")


def main():
    parser = argparse.ArgumentParser(description='GradCAM Visualization for Face SR')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--hr-dir', type=str, default='data/processed/val/HR',
                        help='Directory with HR images')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of images to process')
    parser.add_argument('--output-dir', type=str, default='outputs/gradcam',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.device)

    if args.image:
        # Single image
        process_single_image(model, args.image, output_dir, args.device)
    else:
        # Directory of images
        hr_dir = Path(args.hr_dir)
        image_files = list(hr_dir.glob('*.png')) + list(hr_dir.glob('*.jpg'))
        image_files = sorted(image_files)[:args.num_images]

        print(f"\nProcessing {len(image_files)} images...")

        for img_path in tqdm(image_files, desc="Generating GradCAM"):
            process_single_image(model, str(img_path), output_dir, args.device)

    print(f"\nGradCAM visualizations saved to: {output_dir}/")


if __name__ == '__main__':
    main()
