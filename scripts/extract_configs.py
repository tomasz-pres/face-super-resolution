#!/usr/bin/env python
"""
Extract Training Configs from Checkpoints
==========================================
Creates stage-specific config files from checkpoints.
"""

import torch
import yaml
from pathlib import Path

def extract_config_from_checkpoint(checkpoint_path, stage_name, stage_epochs, loss_config):
    """Extract and create config file from checkpoint."""

    print(f"\nExtracting config from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get saved config if available
    saved_config = checkpoint.get('config', {})

    # Get model architecture from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # Infer model config
    model_config = {
        'num_channels': 64,
        'num_groups': 6,
        'blocks_per_group': 10,
        'reduction_ratio': 4,
        'scale_factor': 4,
        'res_scale': 0.2,
    }

    # Try to infer from state dict
    for key in state_dict.keys():
        if 'conv_first.weight' in key:
            model_config['num_channels'] = state_dict[key].shape[0]
            break

    # Create full config
    config = {
        'experiment_name': f'face_sr_{stage_name}',
        'stage': stage_name,
        'epochs': stage_epochs,

        # Model architecture
        'model': {
            'num_channels': model_config['num_channels'],
            'num_groups': model_config['num_groups'],
            'blocks_per_group': model_config['blocks_per_group'],
            'reduction_ratio': model_config['reduction_ratio'],
            'scale_factor': model_config['scale_factor'],
            'res_scale': model_config['res_scale'],
        },

        # Training parameters
        'training': {
            'batch_size': saved_config.get('batch_size', 48),
            'learning_rate': saved_config.get('learning_rate', 1e-4),
            'weight_decay': saved_config.get('weight_decay', 0),
            'gradient_clip': saved_config.get('gradient_clip', None),
            'accumulation_steps': saved_config.get('accumulation_steps', 1),
            'use_amp': saved_config.get('use_amp', False),
        },

        # Scheduler
        'scheduler': {
            'type': saved_config.get('scheduler_type', 'cosine'),
            'T_max': saved_config.get('scheduler_T_max', 100),
            'eta_min': saved_config.get('scheduler_eta_min', 1e-6),
        },

        # Loss weights (stage-specific)
        'loss_weights': loss_config,

        # Data
        'data': {
            'dataset': 'flickrfaceshq-dataset-nvidia-resized-256px',
            'hr_size': 256,
            'lr_size': 64,
            'scale_factor': 4,
            'augmentation': {
                'horizontal_flip': True,
                'flip_probability': 0.5,
            }
        },

        # Paths
        'paths': {
            'train_hr': 'data/processed/train/HR',
            'val_hr': 'data/processed/val/HR',
            'test_hr': 'data/processed/test/HR',
            'checkpoint_dir': 'checkpoints',
            'output_dir': 'outputs',
        }
    }

    return config


def main():
    """Extract configs for all three stages."""

    stages = [
        {
            'checkpoint': 'checkpoints/big_model_after_epoch_100.pth',
            'name': 'stage1_psnr',
            'epochs': 100,
            'loss_config': {
                'l1': 1.0,
                'perceptual': 1.0,
                'ssim': 0.0,
                'adversarial': 0.0,
                'comment': 'PSNR-oriented pre-training'
            }
        },
        {
            'checkpoint': 'checkpoints/epoch_50.pth',
            'name': 'stage2_ssim',
            'epochs': 50,
            'loss_config': {
                'l1': 1.0,
                'perceptual': 0.5,
                'ssim': 0.2,
                'adversarial': 0.0,
                'comment': 'SSIM fine-tuning (did not bring improvements)'
            }
        },
        {
            'checkpoint': 'checkpoints/final_custom_model.pth',
            'name': 'stage3_gan',
            'epochs': 20,
            'loss_config': {
                'l1': 0.01,
                'perceptual': 1.0,
                'ssim': 0.0,
                'adversarial': 0.005,
                'comment': 'GAN fine-tuning for perceptual quality'
            }
        }
    ]

    configs_dir = Path('configs/stages')
    configs_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EXTRACTING TRAINING STAGE CONFIGS")
    print("="*70)

    for stage_info in stages:
        checkpoint_path = Path(stage_info['checkpoint'])

        if not checkpoint_path.exists():
            print(f"\n[WARNING] {checkpoint_path} not found, skipping...")
            continue

        # Extract config
        config = extract_config_from_checkpoint(
            str(checkpoint_path),
            stage_info['name'],
            stage_info['epochs'],
            stage_info['loss_config']
        )

        # Save config file
        config_file = configs_dir / f"{stage_info['name']}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  [OK] Saved: {config_file}")

    print("\n" + "="*70)
    print("CREATING COMBINED CONFIG")
    print("="*70)

    # Create a combined config showing the full training pipeline
    combined_config = {
        'training_pipeline': {
            'total_epochs': 170,
            'stages': [
                {
                    'stage': 1,
                    'name': 'PSNR Pre-training',
                    'epochs': '0-99 (100 epochs)',
                    'config_file': 'configs/stages/stage1_psnr_config.yaml',
                    'loss_weights': {
                        'l1': 1.0,
                        'perceptual': 1.0,
                        'ssim': 0.0,
                        'adversarial': 0.0,
                    },
                    'learning_rate': '1e-4',
                    'scheduler': 'Cosine Annealing (T_max=100)',
                },
                {
                    'stage': 2,
                    'name': 'SSIM Fine-tuning',
                    'epochs': '100-149 (50 epochs)',
                    'config_file': 'configs/stages/stage2_ssim_config.yaml',
                    'loss_weights': {
                        'l1': 1.0,
                        'perceptual': 0.5,
                        'ssim': 0.2,
                        'adversarial': 0.0,
                    },
                    'learning_rate': '1e-5',
                    'note': 'Did not bring significant improvements',
                },
                {
                    'stage': 3,
                    'name': 'GAN Fine-tuning',
                    'epochs': '150-169 (20 epochs)',
                    'config_file': 'configs/stages/stage3_gan_config.yaml',
                    'loss_weights': {
                        'l1': 0.01,
                        'perceptual': 1.0,
                        'ssim': 0.0,
                        'adversarial': 0.005,
                    },
                    'note': 'Major perceptual quality improvements (LPIPS: 0.0695)',
                }
            ]
        }
    }

    pipeline_config_file = configs_dir / 'training_pipeline.yaml'
    with open(pipeline_config_file, 'w') as f:
        yaml.dump(combined_config, f, default_flow_style=False, sort_keys=False)

    print(f"  [OK] Saved pipeline overview: {pipeline_config_file}")

    print("\n" + "="*70)
    print("All configs extracted successfully!")
    print(f"Config files saved to: {configs_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
