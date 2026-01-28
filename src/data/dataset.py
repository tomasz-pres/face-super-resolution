"""
FFHQ Dataset Class for Super-Resolution
========================================
PyTorch Dataset with support for HDF5 and directory loading.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from collections import OrderedDict
import threading

import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import Dataset

from .transforms import PairedTransform, to_tensor


class ImageCache:
    """LRU cache for frequently accessed images."""

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: Maximum number of images to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Tuple[np.ndarray, np.ndarray]) -> None:
        """Add item to cache."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    self.cache.popitem(last=False)
                self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class FFHQDataset(Dataset):
    """
    FFHQ Dataset for Super-Resolution Training.

    Supports loading from:
    - Directory structure with HR/ and LR/ subdirectories
    - Directory structure with HR/ only (LR generated on-the-fly)
    - HDF5 files with 'HR' and 'LR' datasets

    Features:
    - Synchronized augmentation for LR-HR pairs
    - LRU caching for faster loading
    - Patch-based training support
    - On-the-fly LR generation when LR directory doesn't exist
    """

    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 scale_factor: int = 4,
                 hr_patch_size: int = 128,
                 use_cache: bool = True,
                 cache_size: int = 100,
                 return_filename: bool = False,
                 horizontal_flip: float = 0.5,
                 random_rotate90: float = 0.0,
                 color_jitter_prob: float = 0.3,
                 brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.05,
                 generate_lr_on_the_fly: bool = True):
        """
        Args:
            data_root: Path to data directory or HDF5 file
            mode: 'train', 'val', or 'test'
            scale_factor: Upscaling factor (4 for 64->256)
            hr_patch_size: Size of HR patches for training
            use_cache: Whether to cache images in memory
            cache_size: Maximum number of images to cache
            return_filename: Whether to return filename with each sample
            horizontal_flip: Probability of horizontal flip
            random_rotate90: Probability of 90-degree rotation
            color_jitter_prob: Probability of color jitter
            brightness: Brightness jitter range
            contrast: Contrast jitter range
            saturation: Saturation jitter range
            hue: Hue jitter range
            generate_lr_on_the_fly: If True and LR dir missing, generate LR from HR
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.mode = mode
        self.scale_factor = scale_factor
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale_factor
        self.return_filename = return_filename
        self.generate_lr_on_the_fly = generate_lr_on_the_fly
        self.hr_only_mode = False  # Will be set in _init_directory if needed

        # Determine data source
        self.use_hdf5 = False
        self.h5_file = None

        if self.data_root.suffix == '.h5':
            # HDF5 file
            self.use_hdf5 = True
            self._init_hdf5()
        elif (self.data_root / f'{mode}.h5').exists():
            # HDF5 file in directory
            self.use_hdf5 = True
            self.data_root = self.data_root / f'{mode}.h5'
            self._init_hdf5()
        else:
            # Directory structure
            self._init_directory()

        # Setup transforms with augmentation config
        self.transform = PairedTransform(
            hr_patch_size=hr_patch_size,
            scale_factor=scale_factor,
            mode=mode,
            horizontal_flip=horizontal_flip,
            random_rotate90=random_rotate90,
            color_jitter_prob=color_jitter_prob,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

        # Setup cache
        self.use_cache = use_cache and mode == 'train'
        self.cache = ImageCache(cache_size) if self.use_cache else None

    def _init_hdf5(self) -> None:
        """Initialize HDF5 data source."""
        self.h5_file = h5py.File(self.data_root, 'r')
        self.hr_data = self.h5_file['HR']
        self.lr_data = self.h5_file['LR']

        if 'filenames' in self.h5_file:
            self.filenames = [f.decode() if isinstance(f, bytes) else f
                             for f in self.h5_file['filenames']]
        else:
            self.filenames = [f'{i:05d}.png' for i in range(len(self.hr_data))]

        self.length = len(self.hr_data)

    def _init_directory(self) -> None:
        """Initialize directory data source."""
        # Look for mode subdirectory
        mode_dir = self.data_root / self.mode
        if mode_dir.exists():
            hr_dir = mode_dir / 'HR'
            lr_dir = mode_dir / 'LR'
        else:
            hr_dir = self.data_root / 'HR'
            lr_dir = self.data_root / 'LR'

        if not hr_dir.exists():
            raise ValueError(f"Could not find HR directory in {self.data_root}")

        # Get HR file list
        self.hr_files = sorted(hr_dir.glob('*.png'))
        if not self.hr_files:
            # Also try jpg
            self.hr_files = sorted(hr_dir.glob('*.jpg'))

        if not self.hr_files:
            raise ValueError(f"No images found in {hr_dir}")

        # Check if LR directory exists
        if not lr_dir.exists() or not list(lr_dir.glob('*.png')):
            if self.generate_lr_on_the_fly:
                # HR-only mode: LR will be generated on-the-fly
                self.hr_only_mode = True
                self.lr_files = []
                print(f"HR-only mode: {len(self.hr_files)} HR images, LR generated on-the-fly")
            else:
                raise ValueError(f"Could not find LR directory in {self.data_root} and generate_lr_on_the_fly=False")
        else:
            # Both HR and LR directories exist
            self.lr_files = sorted(lr_dir.glob('*.png'))

            # Verify matching
            hr_names = {f.stem for f in self.hr_files}
            lr_names = {f.stem for f in self.lr_files}

            if hr_names != lr_names:
                missing_lr = hr_names - lr_names
                missing_hr = lr_names - hr_names
                if missing_lr:
                    print(f"Warning: {len(missing_lr)} HR images without LR pair")
                if missing_hr:
                    print(f"Warning: {len(missing_hr)} LR images without HR pair")

                # Keep only matching pairs
                common = hr_names & lr_names
                self.hr_files = [f for f in self.hr_files if f.stem in common]
                self.lr_files = [f for f in self.lr_files if f.stem in common]

        self.filenames = [f.name for f in self.hr_files]
        self.length = len(self.hr_files)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample.

        Returns:
            Dictionary with keys:
            - 'lr': LR tensor (C, H, W)
            - 'hr': HR tensor (C, H, W)
            - 'filename': (optional) Original filename
        """
        # Try cache first
        cache_key = f"{self.mode}_{idx}"
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                hr_image, lr_image = cached
                hr_image = hr_image.copy()
                lr_image = lr_image.copy()
            else:
                hr_image, lr_image = self._load_images(idx)
                self.cache.put(cache_key, (hr_image.copy(), lr_image.copy()))
        else:
            hr_image, lr_image = self._load_images(idx)

        # Apply transforms
        hr_image, lr_image = self.transform(hr_image, lr_image)

        # Convert to tensors
        hr_tensor = to_tensor(hr_image, normalize=True)
        lr_tensor = to_tensor(lr_image, normalize=True)

        result = {
            'lr': lr_tensor,
            'hr': hr_tensor,
        }

        if self.return_filename:
            result['filename'] = self.filenames[idx]

        return result

    def _load_images(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load HR and LR images for given index."""
        if self.use_hdf5:
            hr_image = self.hr_data[idx]
            lr_image = self.lr_data[idx]
        else:
            hr_path = self.hr_files[idx]
            hr_image = cv2.imread(str(hr_path))
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

            if self.hr_only_mode:
                # Generate LR from HR using bicubic downscaling
                h, w = hr_image.shape[:2]
                lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
                lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            else:
                lr_path = self.lr_files[idx]
                lr_image = cv2.imread(str(lr_path))
                lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        return hr_image, lr_image

    def __del__(self):
        """Cleanup HDF5 file handle."""
        if self.h5_file is not None:
            self.h5_file.close()

    def get_sample_images(self, n: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get n random sample image pairs for visualization."""
        indices = np.random.choice(len(self), min(n, len(self)), replace=False)
        samples = []

        for idx in indices:
            hr_image, lr_image = self._load_images(idx)
            samples.append((lr_image, hr_image))

        return samples


def get_dataloader(data_root: str,
                   mode: str = 'train',
                   batch_size: int = 16,
                   num_workers: int = 4,
                   **dataset_kwargs) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the FFHQ dataset.

    Args:
        data_root: Path to data directory
        mode: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for FFHQDataset

    Returns:
        DataLoader instance
    """
    dataset = FFHQDataset(data_root, mode=mode, **dataset_kwargs)

    shuffle = (mode == 'train')
    drop_last = (mode == 'train')

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )


# Convenience function for quick testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data/processed')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_root}...")
    dataset = FFHQDataset(args.data_root, mode=args.mode)
    print(f"Dataset size: {len(dataset)}")

    # Test loading
    sample = dataset[0]
    print(f"LR shape: {sample['lr'].shape}")
    print(f"HR shape: {sample['hr'].shape}")
    print(f"LR range: [{sample['lr'].min():.3f}, {sample['lr'].max():.3f}]")
    print(f"HR range: [{sample['hr'].min():.3f}, {sample['hr'].max():.3f}]")
