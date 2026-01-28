"""
Training Pipeline for Face Super-Resolution
=============================================
Complete training loop with MLOps integration.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torchvision.utils import make_grid, save_image
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def save_validation_grid(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    epoch: int,
    save_dir: str = "training_logs"
) -> None:
    """
    Save a grid visualization comparing LR, SR, and HR images.

    Args:
        lr_images: Low-resolution input images (B, 3, H, W)
        sr_images: Super-resolution output images (B, 3, H, W)
        hr_images: High-resolution ground truth images (B, 3, H, W)
        epoch: Current epoch number
        save_dir: Directory to save the visualization
    """
    if not TORCHVISION_AVAILABLE:
        print("Warning: torchvision not available, skipping visualization")
        return

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Ensure tensors are on CPU and clamped to [0, 1]
    lr_images = lr_images.detach().cpu().clamp(0, 1)
    sr_images = sr_images.detach().cpu().clamp(0, 1)
    hr_images = hr_images.detach().cpu().clamp(0, 1)

    # Resize LR to match HR for visualization
    lr_upscaled = F.interpolate(lr_images, size=hr_images.shape[-2:], mode='nearest')

    # Take first 4 images (or less if batch is smaller)
    num_images = min(4, lr_images.shape[0])

    # Stack: LR (upscaled) | SR | HR for each image
    rows = []
    for i in range(num_images):
        rows.append(lr_upscaled[i])
        rows.append(sr_images[i])
        rows.append(hr_images[i])

    # Create grid (3 rows per image: LR, SR, HR)
    grid = make_grid(rows, nrow=3, padding=2, normalize=False)

    # Save grid
    save_image(grid, save_path / f"epoch_{epoch:04d}.png")
    print(f"Saved validation grid to {save_path / f'epoch_{epoch:04d}.png'}")


@dataclass
class TrainerConfig:
    """Configuration for trainer."""

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_T_max: int = 50
    scheduler_eta_min: float = 1e-7
    scheduler_step_size: int = 10  # For StepLR
    scheduler_gamma: float = 0.5   # For StepLR

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'val_psnr'
    early_stopping_mode: str = 'max'  # 'max' or 'min'

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 10
    save_best: bool = True

    # Logging
    log_every: int = 100
    log_images_every: int = 5
    use_wandb: bool = True
    wandb_project: str = 'face-super-resolution'

    # Device
    device: str = 'cuda'

    # GAN Training
    gan_weight: float = 0.0  # 0 = no GAN, > 0 = enable adversarial training
    gan_type: str = 'vanilla'  # 'vanilla', 'lsgan', 'wgan'
    d_learning_rate: float = 1e-4  # Discriminator learning rate
    d_weight_decay: float = 0.0
    d_updates_per_g: int = 1  # D updates per G update
    gan_start_epoch: int = 0  # Epoch to start GAN training (warmup)


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, mode: str = 'max',
                 min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Training manager for super-resolution models.

    Features:
    - Mixed precision training
    - Gradient clipping and accumulation
    - W&B logging
    - Checkpoint management
    - Early stopping
    - Training dynamics tracking
    - GAN adversarial training support
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 config: Optional[TrainerConfig] = None,
                 discriminator: Optional[nn.Module] = None,
                 gan_loss: Optional[nn.Module] = None):
        """
        Args:
            model: Generator model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function (L1 + Perceptual + SSIM)
            config: Trainer configuration
            discriminator: Optional discriminator for GAN training
            gan_loss: Optional GAN loss module
        """
        self.config = config or TrainerConfig()

        # Set device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )

        # Generator model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss - move to device
        self.loss_fn = loss_fn.to(self.device)

        # Generator optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None

        # GAN components
        self.use_gan = self.config.gan_weight > 0 and discriminator is not None
        self.discriminator = None
        self.optimizer_d = None
        self.gan_loss = None
        self.scaler_d = None

        if self.use_gan:
            print(f"GAN training enabled with weight={self.config.gan_weight}")
            self.discriminator = discriminator.to(self.device)
            self.gan_loss = gan_loss.to(self.device) if gan_loss else self._create_gan_loss()

            # Discriminator optimizer
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.d_learning_rate,
                weight_decay=self.config.d_weight_decay
            )

            # Separate scaler for discriminator (recommended for stability)
            self.scaler_d = GradScaler() if self.config.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode=self.config.early_stopping_mode
        )

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history: Dict[str, List] = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'learning_rate': [],
        }

        # Add GAN-specific history if using GAN
        if self.use_gan:
            self.training_history['d_loss'] = []
            self.training_history['g_loss'] = []
            self.training_history['d_real'] = []
            self.training_history['d_fake'] = []

        # W&B
        self.use_wandb = self.config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()

    def _create_gan_loss(self):
        """Create default GAN loss based on config."""
        from src.models.discriminator import GANLoss
        return GANLoss(gan_type=self.config.gan_type)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler_T_max,
                eta_min=self.config.scheduler_eta_min
            )
        elif self.config.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            return None

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        wandb.init(
            project=self.config.wandb_project,
            config={
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'gradient_clip': self.config.gradient_clip,
                'use_amp': self.config.use_amp,
                'model': self.model.__class__.__name__,
            }
        )
        wandb.watch(self.model, log='gradients', log_freq=100)

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation epoch
            val_metrics = self._validate_epoch()

            # Update scheduler
            if self.scheduler is not None:
                if self.config.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['psnr'])
                else:
                    self.scheduler.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pth')

            # Save best model
            if self.config.save_best:
                metric_value = val_metrics.get(
                    self.config.early_stopping_metric.replace('val_', ''),
                    val_metrics.get('psnr', 0)
                )
                if self._is_best(metric_value):
                    self._save_checkpoint('best_model.pth', is_best=True)

            # Early stopping - use configured metric
            stop_metric = val_metrics.get(
                self.config.early_stopping_metric.replace('val_', ''),
                val_metrics.get('psnr', 0)
            )
            if self.early_stopping(stop_metric):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Final save
        self._save_checkpoint('final_model.pth')

        if self.use_wandb:
            wandb.finish()

        return self.training_history

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch (supports both standard and GAN training)."""
        self.model.train()
        if self.use_gan:
            self.discriminator.train()

        # Check if GAN should be active this epoch
        gan_active = self.use_gan and self.current_epoch >= self.config.gan_start_epoch

        total_loss = 0.0
        total_d_loss = 0.0
        total_g_adv_loss = 0.0
        total_d_real = 0.0
        total_d_fake = 0.0
        loss_components = {}
        num_batches = 0

        desc = f'Epoch {self.current_epoch + 1}'
        if gan_active:
            desc += ' [GAN]'
        pbar = tqdm(self.train_loader, desc=desc)

        for batch_idx, batch in enumerate(pbar):
            hr_images = batch['hr'].to(self.device)

            # Generate LR on-the-fly for perfect alignment with model's bicubic skip
            lr_images = F.interpolate(
                hr_images,
                scale_factor=0.25,
                mode='bicubic',
                align_corners=False
            )

            # ============== GAN Training ==============
            if gan_active:
                # ---------- Update Discriminator ----------
                for _ in range(self.config.d_updates_per_g):
                    self.optimizer_d.zero_grad()

                    with autocast(enabled=self.config.use_amp):
                        # Generate fake images (detached from generator graph)
                        with torch.no_grad():
                            sr_images_d = self.model(lr_images)

                        # Discriminator predictions
                        d_real = self.discriminator(hr_images)
                        d_fake = self.discriminator(sr_images_d.detach())

                        # Discriminator loss
                        d_loss_real = self.gan_loss(d_real, is_real=True)
                        d_loss_fake = self.gan_loss(d_fake, is_real=False)
                        d_loss = (d_loss_real + d_loss_fake) / 2

                    # Backward pass for discriminator
                    if self.scaler_d is not None:
                        self.scaler_d.scale(d_loss).backward()
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                    else:
                        d_loss.backward()
                        self.optimizer_d.step()

                # Track D metrics
                total_d_loss += d_loss.item()
                total_d_real += torch.sigmoid(d_real).mean().item()
                total_d_fake += torch.sigmoid(d_fake).mean().item()

            # ---------- Update Generator ----------
            self.optimizer.zero_grad()

            with autocast(enabled=self.config.use_amp):
                # Forward pass
                sr_images = self.model(lr_images)

                # Content loss (L1 + Perceptual + SSIM)
                content_loss, components = self.loss_fn(sr_images, hr_images)

                # Add adversarial loss if GAN is active
                if gan_active:
                    d_fake_for_g = self.discriminator(sr_images)
                    g_adv_loss = self.gan_loss(d_fake_for_g, is_real=True)  # Generator wants D to think it's real
                    total_g_adv_loss += g_adv_loss.item()

                    # Total generator loss
                    loss = content_loss + self.config.gan_weight * g_adv_loss
                    components['g_adv'] = g_adv_loss
                else:
                    loss = content_loss

                loss = loss / self.config.accumulation_steps

            # Backward pass for generator
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.global_step += 1

            # Track metrics
            total_loss += loss.item() * self.config.accumulation_steps
            for name, value in components.items():
                if name not in loss_components:
                    loss_components[name] = 0.0
                loss_components[name] += value.item()
            num_batches += 1

            # Update progress bar
            postfix = {'loss': total_loss / num_batches}
            if gan_active:
                postfix['D'] = total_d_loss / num_batches
                postfix['D_real'] = total_d_real / num_batches
                postfix['D_fake'] = total_d_fake / num_batches
            pbar.set_postfix(postfix)

            # Log to W&B
            if self.use_wandb and self.global_step % self.config.log_every == 0:
                log_dict = {
                    'train/loss': loss.item() * self.config.accumulation_steps,
                    'train/step': self.global_step,
                    **{f'train/{k}': v.item() for k, v in components.items()}
                }
                if gan_active:
                    log_dict.update({
                        'train/d_loss': d_loss.item(),
                        'train/d_real': torch.sigmoid(d_real).mean().item(),
                        'train/d_fake': torch.sigmoid(d_fake).mean().item(),
                    })
                wandb.log(log_dict)

        # Compute averages
        metrics = {'loss': total_loss / num_batches}
        for name, value in loss_components.items():
            metrics[name] = value / num_batches

        # Add GAN metrics
        if gan_active:
            metrics['d_loss'] = total_d_loss / num_batches
            metrics['g_adv_loss'] = total_g_adv_loss / num_batches
            metrics['d_real'] = total_d_real / num_batches
            metrics['d_fake'] = total_d_fake / num_batches

        return metrics

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        sample_images = []

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
            hr_images = batch['hr'].to(self.device)

            # Generate LR on-the-fly for perfect alignment
            lr_images = F.interpolate(
                hr_images,
                scale_factor=0.25,
                mode='bicubic',
                align_corners=False
            )

            # Forward pass
            with torch.no_grad():
                sr_images = self.model(lr_images)
            loss, _ = self.loss_fn(sr_images, hr_images)

            # Compute metrics
            psnr = self._compute_psnr(sr_images, hr_images)
            ssim = self._compute_ssim(sr_images, hr_images)

            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1

            # Collect sample images for visualization (first batch only)
            if batch_idx == 0:
                sample_lr = lr_images[:8].clone()
                sample_sr = sr_images[:8].clone()
                sample_hr = hr_images[:8].clone()

        metrics = {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches,
        }

        # Save validation grid every epoch
        if 'sample_lr' in locals():
            save_validation_grid(
                sample_lr, sample_sr, sample_hr,
                epoch=self.current_epoch,
                save_dir="training_logs"
            )

        # Log images to W&B
        if self.use_wandb and self.current_epoch % self.config.log_images_every == 0:
            sample_images = [
                sample_lr[0].cpu() if 'sample_lr' in locals() else None,
                sample_sr[0].cpu() if 'sample_sr' in locals() else None,
                sample_hr[0].cpu() if 'sample_hr' in locals() else None,
            ]
            if all(s is not None for s in sample_images):
                self._log_images(sample_images)

        return metrics

    def _compute_psnr(self, pred: torch.Tensor,
                      target: torch.Tensor) -> float:
        """Compute PSNR metric."""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()

    def _compute_ssim(self, pred: torch.Tensor,
                      target: torch.Tensor) -> float:
        """Compute SSIM metric (simplified)."""
        from src.losses.ssim_loss import ssim
        return ssim(pred, target).item()

    def _log_epoch_metrics(self, epoch: int,
                           train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float],
                           lr: float) -> None:
        """Log epoch metrics."""
        # Update history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_psnr'].append(val_metrics['psnr'])
        self.training_history['val_ssim'].append(val_metrics['ssim'])
        self.training_history['learning_rate'].append(lr)

        # Print
        print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val PSNR:   {val_metrics['psnr']:.2f} dB")
        print(f"  Val SSIM:   {val_metrics['ssim']:.4f}")
        print(f"  LR:         {lr:.2e}")

        # W&B
        if self.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_metrics['loss'],
                'val/loss': val_metrics['loss'],
                'val/psnr': val_metrics['psnr'],
                'val/ssim': val_metrics['ssim'],
                'learning_rate': lr,
            })

    def _log_images(self, images: List[torch.Tensor]) -> None:
        """Log sample images to W&B."""
        if not self.use_wandb:
            return

        lr, sr, hr = images

        # Convert to numpy
        lr_np = lr.permute(1, 2, 0).numpy()
        sr_np = sr.permute(1, 2, 0).numpy()
        hr_np = hr.permute(1, 2, 0).numpy()

        wandb.log({
            'samples/lr': wandb.Image(lr_np, caption='Low Resolution'),
            'samples/sr': wandb.Image(sr_np, caption='Super Resolution'),
            'samples/hr': wandb.Image(hr_np, caption='High Resolution'),
        })

    def _is_best(self, metric_value: float) -> bool:
        """Check if current metric is best."""
        if self.best_metric is None:
            self.best_metric = metric_value
            return True

        if self.config.early_stopping_mode == 'max':
            is_better = metric_value > self.best_metric
        else:
            is_better = metric_value < self.best_metric

        if is_better:
            self.best_metric = metric_value

        return is_better

    def _save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config.__dict__,
        }

        # Save discriminator state if using GAN
        if self.use_gan and self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            print(f"  New best model saved: {self.best_metric:.4f}")

    def load_checkpoint(self, path: str, weights_only: bool = False) -> None:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            weights_only: If True, only load model weights (for fine-tuning).
                          Optimizer, scheduler, and epoch are NOT restored.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if weights_only:
            # Fine-tuning mode: only load weights, start fresh training
            print(f"Loaded model weights from epoch {checkpoint['epoch']} (fine-tuning mode)")
            print(f"  Starting fresh with LR={self.config.learning_rate}")
        else:
            # Full resume: restore optimizer, scheduler, epoch
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load discriminator state if available and using GAN
            if self.use_gan and 'discriminator_state_dict' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                if 'optimizer_d_state_dict' in checkpoint:
                    self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                print("  Loaded discriminator state")

            self.current_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_metric = checkpoint['best_metric']
            self.training_history = checkpoint['training_history']

            print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")


def overfit_test(model: nn.Module,
                 dataloader: DataLoader,
                 loss_fn: nn.Module,
                 num_images: int = 10,
                 num_iterations: int = 1000,
                 device: str = 'cuda') -> Dict[str, Any]:
    """
    Test model's ability to overfit on small batch.

    Args:
        model: Model to test
        dataloader: Data loader
        loss_fn: Loss function
        num_images: Number of images to overfit
        num_iterations: Training iterations
        device: Device to use

    Returns:
        Results dictionary
    """
    print(f"\nOverfitting test on {num_images} images...")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    # Use pure MSE for overfitting test (directly optimizes PSNR)
    mse_loss = nn.MSELoss()

    # Get HR images only from dataloader
    batch = next(iter(dataloader))
    hr_images = batch['hr'][:num_images].to(device)

    # GENERATE LR from HR using PyTorch (ensures perfect alignment)
    # This bypasses any disk-based LR/HR mismatch issues
    lr_images = F.interpolate(
        hr_images,
        scale_factor=1/4,
        mode='bicubic',
        align_corners=False
    )

    print(f"HR shape: {hr_images.shape}, LR shape: {lr_images.shape}")
    print(f"HR range: [{hr_images.min():.3f}, {hr_images.max():.3f}]")
    print(f"LR range: [{lr_images.min():.3f}, {lr_images.max():.3f}]")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    losses = []
    psnrs = []

    pbar = tqdm(range(num_iterations), desc='Overfitting')
    for i in pbar:
        optimizer.zero_grad()

        sr_images = model(lr_images)
        sr_images = torch.clamp(sr_images, 0.0, 1.0)  # Clamp before loss
        loss = mse_loss(sr_images, hr_images)

        loss.backward()
        optimizer.step()

        # Compute PSNR
        with torch.no_grad():
            mse = torch.mean((sr_images - hr_images) ** 2)
            psnr = 10 * torch.log10(1.0 / mse).item()

        losses.append(loss.item())
        psnrs.append(psnr)

        pbar.set_postfix({'loss': loss.item(), 'psnr': psnr})

    results = {
        'final_loss': losses[-1],
        'final_psnr': psnrs[-1],
        'loss_history': losses,
        'psnr_history': psnrs,
        'converged': psnrs[-1] > 35,  # Should be very high for overfitting
    }

    print(f"\nOverfit test results:")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Final PSNR: {results['final_psnr']:.2f} dB")
    print(f"  Converged: {results['converged']}")

    return results
