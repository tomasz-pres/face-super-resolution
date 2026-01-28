#!/usr/bin/env python
"""
Training Script for Face Super-Resolution
==========================================
Main entry point for training models.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --model custom --epochs 50
"""

import argparse
import sys
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch

from src.data import FFHQDataset, get_dataloader
from src.models import (
    create_face_enhance_net,
    create_transfer_model,
    create_esrgan_baseline,
    FaceEnhanceNetConfig
)
from src.models.discriminator import create_discriminator, GANLoss
from src.losses import create_loss_function
from src.training import Trainer, TrainerConfig, overfit_test


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def create_model(model_type: str, config: dict):
    """Create model based on type."""
    if model_type == 'custom':
        model_config = config.get('model', {}).get('custom', {})
        return create_face_enhance_net(
            num_channels=model_config.get('num_channels', 64),
            num_groups=model_config.get('num_groups', 3),
            blocks_per_group=model_config.get('blocks_per_group', 4),
            reduction_ratio=model_config.get('reduction_ratio', 4),
            scale_factor=model_config.get('upscale_factor', 4),
            res_scale=model_config.get('res_scale', 0.2)
        )
    elif model_type == 'transfer':
        transfer_config = config.get('model', {}).get('transfer', {})
        return create_transfer_model(
            backbone_blocks=transfer_config.get('backbone_blocks', 16),
            freeze_blocks=transfer_config.get('freeze_blocks', 16),
            head_blocks=transfer_config.get('head_blocks', 4),
            head_channels=transfer_config.get('head_channels', 64),
            scale_factor=transfer_config.get('scale_factor', 4)
        )
    elif model_type == 'esrgan':
        return create_esrgan_baseline()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train Face Super-Resolution Model')

    # Config
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    # Model
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'transfer', 'esrgan'],
                        help='Model type to train')

    # Data
    parser.add_argument('--data-root', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')

    # Training
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--perceptual-weight', type=float, default=None,
                        help='Perceptual loss weight')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience')

    # Resume / Fine-tune
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune mode: load weights only, reset optimizer/scheduler')

    # Overfit test
    parser.add_argument('--overfit-test', action='store_true',
                        help='Run overfitting test before training')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # W&B
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")

    # Extract config sections
    project_config = config.get('project', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    loss_config = config.get('loss', {})
    checkpoint_config = config.get('checkpoint', {})
    logging_config = config.get('logging', {})

    # Set seed for reproducibility
    seed = project_config.get('seed', 42)
    set_seed(seed)

    # Override config with command line arguments (CLI args take priority)
    batch_size = args.batch_size or data_config.get('batch_size', 16)
    epochs = args.epochs or training_config.get('epochs', 50)
    lr = args.lr or training_config.get('optimizer', {}).get('lr', 1e-4)
    data_root = args.data_root or data_config.get('data_root', 'data/processed')
    device = args.device or project_config.get('device', 'cuda')
    model_type = args.model or config.get('model', {}).get('type', 'custom')

    print(f"\n{'='*60}")
    print("Face Super-Resolution Training")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"{'='*60}\n")

    # Create data loaders
    print("Creating data loaders...")

    # Extract augmentation config (at root level, not under data)
    aug_config = config.get('augmentation', {})
    color_jitter = aug_config.get('color_jitter', {})

    train_loader = get_dataloader(
        data_root,
        mode='train',
        batch_size=batch_size,
        num_workers=data_config.get('num_workers', 4),
        hr_patch_size=aug_config.get('random_crop', {}).get('hr_patch_size', 128),
        horizontal_flip=aug_config.get('horizontal_flip', 0.5),
        random_rotate90=aug_config.get('random_rotate90', 0.0),  # 0 for faces!
        color_jitter_prob=color_jitter.get('probability', 0.3),
        brightness=color_jitter.get('brightness', 0.1),
        contrast=color_jitter.get('contrast', 0.1),
        saturation=color_jitter.get('saturation', 0.0),
        hue=color_jitter.get('hue', 0.0)
    )

    val_loader = get_dataloader(
        data_root,
        mode='val',
        batch_size=batch_size,
        num_workers=data_config.get('num_workers', 4)
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, config)

    # Get model info
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"Model parameters: {info.get('total_params', 'N/A'):,}")

    # Create loss function
    print("\nCreating loss function...")
    perceptual_weight = args.perceptual_weight if args.perceptual_weight is not None else loss_config.get('perceptual_weight', 0.01)
    use_charbonnier = loss_config.get('use_charbonnier', False)
    charbonnier_eps = loss_config.get('charbonnier_eps', 1e-3)
    perceptual_layers = loss_config.get('perceptual', {}).get('layers', ['conv3_4', 'conv4_4'])
    ssim_weight = loss_config.get('ssim_weight', 0.1)

    loss_fn = create_loss_function(
        l1_weight=loss_config.get('l1_weight', 1.0),
        perceptual_weight=perceptual_weight,
        ssim_weight=ssim_weight,
        use_charbonnier=use_charbonnier,
        charbonnier_eps=charbonnier_eps,
        perceptual_layers=perceptual_layers
    )
    print(f"L1 weight: {loss_config.get('l1_weight', 1.0)}")
    print(f"Perceptual weight: {perceptual_weight}")
    print(f"Perceptual layers: {perceptual_layers}")
    print(f"SSIM weight: {ssim_weight}")
    print(f"Using Charbonnier loss: {use_charbonnier}")

    # Run overfit test if requested
    if args.overfit_test:
        print("\n" + "="*60)
        print("Running overfitting test...")
        print("="*60)

        overfit_results = overfit_test(
            model, train_loader, loss_fn,
            num_images=10,
            num_iterations=1000,
            device=device
        )

        if not overfit_results['converged']:
            print("\nWarning: Model did not converge on small batch!")
            print("This might indicate an architecture or learning rate issue.")
            response = input("Continue with training? [y/N] ")
            if response.lower() != 'y':
                print("Training aborted.")
                return

    # Create trainer config
    gradient_clip = args.gradient_clip if args.gradient_clip is not None else training_config.get('gradient_clip', 1.0)
    early_stopping_config = training_config.get('early_stopping', {})
    patience = args.patience if args.patience is not None else early_stopping_config.get('patience', 10)

    # Scheduler config
    scheduler_config = training_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    scheduler_T_max = scheduler_config.get('T_max', epochs)
    scheduler_eta_min = scheduler_config.get('eta_min', 1e-7)
    scheduler_step_size = scheduler_config.get('step_size', 10)
    scheduler_gamma = scheduler_config.get('gamma', 0.5)

    print(f"\nTrainer configuration:")
    print(f"  Gradient clip: {gradient_clip}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Scheduler type: {scheduler_type}")
    if scheduler_type == 'step':
        print(f"  Scheduler step_size: {scheduler_step_size}")
        print(f"  Scheduler gamma: {scheduler_gamma}")

    # Logging config
    wandb_config = logging_config.get('wandb', {})
    console_config = logging_config.get('console', {})

    # Determine if wandb should be used
    use_wandb = not args.no_wandb and wandb_config.get('enabled', False)

    # GAN config
    gan_config = loss_config.get('gan', {})
    gan_weight = gan_config.get('weight', 0.0)
    gan_type = gan_config.get('type', 'vanilla')
    d_learning_rate = gan_config.get('d_lr', 1e-4)
    d_weight_decay = gan_config.get('d_weight_decay', 0.0)
    d_updates_per_g = gan_config.get('d_updates_per_g', 1)
    gan_start_epoch = gan_config.get('start_epoch', 0)

    trainer_config = TrainerConfig(
        # Training
        epochs=epochs,
        learning_rate=lr,
        weight_decay=training_config.get('optimizer', {}).get('weight_decay', 0.0),
        gradient_clip=gradient_clip,
        accumulation_steps=training_config.get('accumulation_steps', 1),
        # Mixed precision
        use_amp=training_config.get('mixed_precision', True),
        # Scheduler
        scheduler_type=scheduler_type,
        scheduler_T_max=scheduler_T_max,
        scheduler_eta_min=scheduler_eta_min,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        # Early stopping
        early_stopping_patience=patience,
        early_stopping_metric=early_stopping_config.get('metric', 'val_psnr'),
        early_stopping_mode=early_stopping_config.get('mode', 'max'),
        # Checkpointing
        checkpoint_dir=checkpoint_config.get('save_dir', 'checkpoints'),
        save_every=checkpoint_config.get('save_every', 10),
        save_best=checkpoint_config.get('save_best', True),
        # Logging
        log_every=console_config.get('log_every', 100),
        log_images_every=wandb_config.get('log_images_every', 5),
        use_wandb=use_wandb,
        wandb_project=wandb_config.get('project', 'face-super-resolution'),
        # Device
        device=device,
        # GAN
        gan_weight=gan_weight,
        gan_type=gan_type,
        d_learning_rate=d_learning_rate,
        d_weight_decay=d_weight_decay,
        d_updates_per_g=d_updates_per_g,
        gan_start_epoch=gan_start_epoch,
    )

    print(f"  Accumulation steps: {trainer_config.accumulation_steps}")
    print(f"  Mixed precision: {trainer_config.use_amp}")
    print(f"  W&B logging: {use_wandb}")

    # Create discriminator if GAN training is enabled
    discriminator = None
    gan_loss = None
    if gan_weight > 0:
        print(f"\nGAN Training Configuration:")
        print(f"  GAN weight: {gan_weight}")
        print(f"  GAN type: {gan_type}")
        print(f"  Discriminator LR: {d_learning_rate}")
        print(f"  D updates per G: {d_updates_per_g}")
        print(f"  GAN start epoch: {gan_start_epoch}")

        discriminator = create_discriminator(
            input_size=data_config.get('hr_size', 256),
            base_channels=gan_config.get('d_channels', 64),
            use_bn=gan_config.get('d_use_bn', True)
        )
        gan_loss = GANLoss(gan_type=gan_type)

        d_info = discriminator.get_model_info()
        print(f"  Discriminator params: {d_info['total_params']:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=trainer_config,
        discriminator=discriminator,
        gan_loss=gan_loss
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        trainer.load_checkpoint(args.resume, weights_only=args.fine_tune)

    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    try:
        history = trainer.train()

        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"\nFinal metrics:")
        print(f"  Best PSNR: {max(history['val_psnr']):.2f} dB")
        print(f"  Best SSIM: {max(history['val_ssim']):.4f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer._save_checkpoint('interrupted.pth')
        print("Checkpoint saved to checkpoints/interrupted.pth")


if __name__ == '__main__':
    main()
