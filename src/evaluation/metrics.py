"""
Evaluation Metrics for Super-Resolution
========================================
PSNR, SSIM, LPIPS, and FID metrics implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Import SSIM from our losses module
from src.losses.ssim_loss import ssim as compute_ssim


def psnr(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        data_range: Range of pixel values

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr_val = 10 * torch.log10(data_range ** 2 / mse)
    return psnr_val


def psnr_batch(pred: torch.Tensor, target: torch.Tensor,
               data_range: float = 1.0) -> torch.Tensor:
    """
    Compute PSNR for each image in batch.

    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        data_range: Range of pixel values

    Returns:
        PSNR values (B,)
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr_vals = 10 * torch.log10(data_range ** 2 / (mse + 1e-10))
    return psnr_vals


class PSNR(nn.Module):
    """PSNR metric module."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return psnr(pred, target, self.data_range)


class SSIM(nn.Module):
    """SSIM metric module."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return compute_ssim(pred, target, window_size=self.window_size,
                           data_range=self.data_range)


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity.

    Uses AlexNet or VGG features for perceptual similarity.
    """

    def __init__(self, net: str = 'alex', verbose: bool = False):
        """
        Args:
            net: Network to use ('alex' or 'vgg')
            verbose: Print loading info
        """
        super().__init__()
        self.net = net

        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net, verbose=verbose)
            self.available = True
        except ImportError:
            print("Warning: lpips not installed. LPIPS metric unavailable.")
            self.available = False
            self.loss_fn = None

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance.

        Args:
            pred: Predicted image (B, 3, H, W) in [-1, 1] or [0, 1]
            target: Target image (B, 3, H, W)

        Returns:
            LPIPS distance (lower is better)
        """
        if not self.available:
            return torch.tensor(0.0, device=pred.device)

        # Convert from [0, 1] to [-1, 1] if needed
        if pred.min() >= 0:
            pred = pred * 2 - 1
            target = target * 2 - 1

        return self.loss_fn(pred, target).mean()


class MetricCalculator:
    """
    Comprehensive metric calculator for evaluation.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device for computation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.psnr = PSNR().to(self.device)
        self.ssim = SSIM().to(self.device)
        self.lpips = LPIPS().to(self.device)

    @torch.no_grad()
    def compute_metrics(self, pred: torch.Tensor,
                        target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all metrics for a batch.

        Args:
            pred: Predicted images (B, 3, H, W)
            target: Target images (B, 3, H, W)

        Returns:
            Dictionary of metric values
        """
        pred = pred.to(self.device)
        target = target.to(self.device)

        metrics = {
            'psnr': self.psnr(pred, target).item(),
            'ssim': self.ssim(pred, target).item(),
        }

        if self.lpips.available:
            metrics['lpips'] = self.lpips(pred, target).item()

        return metrics

    @torch.no_grad()
    def evaluate_dataset(self, model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         desc: str = 'Evaluating') -> Dict[str, float]:
        """
        Evaluate model on entire dataset.

        Args:
            model: Model to evaluate
            dataloader: Data loader
            desc: Progress bar description

        Returns:
            Average metrics over dataset
        """
        from tqdm import tqdm

        model = model.to(self.device)
        model.eval()

        all_psnr = []
        all_ssim = []
        all_lpips = []

        for batch in tqdm(dataloader, desc=desc):
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)

            sr = model(lr)
            sr = torch.clamp(sr, 0, 1)

            # Batch metrics
            psnr_vals = psnr_batch(sr, hr)
            all_psnr.extend(psnr_vals.cpu().tolist())

            ssim_val = self.ssim(sr, hr)
            all_ssim.append(ssim_val.item())

            if self.lpips.available:
                lpips_val = self.lpips(sr, hr)
                all_lpips.append(lpips_val.item())

        results = {
            'psnr_mean': np.mean(all_psnr),
            'psnr_std': np.std(all_psnr),
            'ssim_mean': np.mean(all_ssim),
            'ssim_std': np.std(all_ssim),
        }

        if all_lpips:
            results['lpips_mean'] = np.mean(all_lpips)
            results['lpips_std'] = np.std(all_lpips)

        return results


def compute_fid(real_images: List[np.ndarray],
                fake_images: List[np.ndarray],
                device: str = 'cuda') -> float:
    """
    Compute Fréchet Inception Distance.

    Args:
        real_images: List of real images (H, W, 3) uint8
        fake_images: List of generated images
        device: Device for computation

    Returns:
        FID score (lower is better)
    """
    try:
        from pytorch_fid import fid_score
        import tempfile
        import os

        # Save images to temporary directories
        with tempfile.TemporaryDirectory() as real_dir:
            with tempfile.TemporaryDirectory() as fake_dir:
                import cv2

                for i, img in enumerate(real_images):
                    cv2.imwrite(os.path.join(real_dir, f'{i:05d}.png'),
                               cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                for i, img in enumerate(fake_images):
                    cv2.imwrite(os.path.join(fake_dir, f'{i:05d}.png'),
                               cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                fid = fid_score.calculate_fid_given_paths(
                    [real_dir, fake_dir],
                    batch_size=50,
                    device=device,
                    dims=2048
                )

        return fid

    except ImportError:
        print("Warning: pytorch_fid not installed. FID computation unavailable.")
        return -1.0


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")

    pred = torch.rand(4, 3, 256, 256)
    target = torch.rand(4, 3, 256, 256)

    # PSNR
    psnr_val = psnr(pred, target)
    print(f"PSNR: {psnr_val.item():.2f} dB")

    # SSIM
    ssim_val = compute_ssim(pred, target)
    print(f"SSIM: {ssim_val.item():.4f}")

    # Metric calculator
    calc = MetricCalculator(device='cpu')
    metrics = calc.compute_metrics(pred, target)
    print(f"All metrics: {metrics}")

    print("Success!")
