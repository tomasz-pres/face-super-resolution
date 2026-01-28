"""
FFHQ Dataset Preparation Script
================================
Prepares LR-HR image pairs from FFHQ dataset for super-resolution training.

Usage:
    python src/data/prepare_data.py --input data/raw/ffhq --output data/processed
"""

import os
import argparse
import random
import json
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm
import h5py


def create_lr_image(hr_image: np.ndarray, lr_size: int = 64,
                    method: str = 'bicubic') -> np.ndarray:
    """
    Downsample HR image to create LR input.

    Args:
        hr_image: High-resolution image (H, W, C)
        lr_size: Target low-resolution size
        method: Degradation method ('bicubic', 'bilinear', 'realistic')

    Returns:
        Low-resolution image
    """
    if method == 'bicubic':
        # Standard bicubic downsampling
        lr = cv2.resize(hr_image, (lr_size, lr_size),
                       interpolation=cv2.INTER_CUBIC)

    elif method == 'bilinear':
        # Bilinear downsampling
        lr = cv2.resize(hr_image, (lr_size, lr_size),
                       interpolation=cv2.INTER_LINEAR)

    elif method == 'realistic':
        # More realistic degradation (blur + noise + downsample)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(hr_image, (7, 7), 1.5)
        # Add Gaussian noise
        noise = np.random.normal(0, 5, blurred.shape).astype(np.float32)
        noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Downsample
        lr = cv2.resize(noisy, (lr_size, lr_size),
                       interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"Unknown degradation method: {method}")

    return lr


def resize_hr_image(image: np.ndarray, hr_size: int = 256) -> np.ndarray:
    """
    Resize image to target HR size using high-quality downsampling.

    Args:
        image: Original image
        hr_size: Target high-resolution size

    Returns:
        Resized image
    """
    # Use INTER_AREA for high-quality downsampling
    return cv2.resize(image, (hr_size, hr_size), interpolation=cv2.INTER_AREA)


def get_image_files(input_dir: Path) -> List[Path]:
    """Get all image files from directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = []

    for ext in extensions:
        files.extend(input_dir.glob(f'*{ext}'))
        files.extend(input_dir.glob(f'*{ext.upper()}'))

    # Also search subdirectories
    for ext in extensions:
        files.extend(input_dir.glob(f'**/*{ext}'))
        files.extend(input_dir.glob(f'**/*{ext.upper()}'))

    # Remove duplicates and sort
    files = sorted(set(files))
    return files


def split_dataset(files: List[Path], train_ratio: float = 0.857,
                  val_ratio: float = 0.071, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split files into train/val/test sets.

    Args:
        files: List of file paths
        train_ratio: Ratio for training set (~60000/70000)
        val_ratio: Ratio for validation set (~5000/70000)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def process_and_save_images(files: List[Path], output_dir: Path,
                            hr_size: int = 256, lr_size: int = 64,
                            degradation: str = 'bicubic',
                            desc: str = "Processing") -> int:
    """
    Process images and save LR-HR pairs.

    Args:
        files: List of input file paths
        output_dir: Output directory
        hr_size: Target HR size
        lr_size: Target LR size
        degradation: Degradation method
        desc: Progress bar description

    Returns:
        Number of successfully processed images
    """
    hr_dir = output_dir / 'HR'
    lr_dir = output_dir / 'LR'
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = []

    for file_path in tqdm(files, desc=desc):
        try:
            # Read image
            img = cv2.imread(str(file_path))
            if img is None:
                errors.append(f"Could not read: {file_path}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create HR image
            hr_img = resize_hr_image(img, hr_size)

            # Create LR image
            lr_img = create_lr_image(hr_img, lr_size, degradation)

            # Generate output filename
            out_name = f"{file_path.stem}.png"

            # Save images (convert back to BGR for OpenCV)
            cv2.imwrite(str(hr_dir / out_name), cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(lr_dir / out_name), cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))

            processed += 1

        except Exception as e:
            errors.append(f"Error processing {file_path}: {str(e)}")

    # Log errors if any
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return processed


def save_to_hdf5(output_dir: Path, split: str, hr_size: int = 256,
                 lr_size: int = 64) -> None:
    """
    Convert processed images to HDF5 format for efficient loading.

    Args:
        output_dir: Directory containing HR and LR folders
        split: Split name (train, val, test)
        hr_size: HR image size
        lr_size: LR image size
    """
    hr_dir = output_dir / 'HR'
    lr_dir = output_dir / 'LR'

    hr_files = sorted(hr_dir.glob('*.png'))

    if not hr_files:
        print(f"No images found in {hr_dir}")
        return

    h5_path = output_dir / f'{split}.h5'

    with h5py.File(h5_path, 'w') as f:
        # Create datasets with compression
        n_images = len(hr_files)
        hr_dataset = f.create_dataset(
            'HR', shape=(n_images, hr_size, hr_size, 3),
            dtype=np.uint8, compression='gzip', compression_opts=4
        )
        lr_dataset = f.create_dataset(
            'LR', shape=(n_images, lr_size, lr_size, 3),
            dtype=np.uint8, compression='gzip', compression_opts=4
        )

        # Store filenames
        dt = h5py.special_dtype(vlen=str)
        filenames = f.create_dataset('filenames', (n_images,), dtype=dt)

        for i, hr_file in enumerate(tqdm(hr_files, desc=f"Creating {split}.h5")):
            lr_file = lr_dir / hr_file.name

            hr_img = cv2.imread(str(hr_file))
            lr_img = cv2.imread(str(lr_file))

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

            hr_dataset[i] = hr_img
            lr_dataset[i] = lr_img
            filenames[i] = hr_file.name

        # Add metadata
        f.attrs['hr_size'] = hr_size
        f.attrs['lr_size'] = lr_size
        f.attrs['n_images'] = n_images
        f.attrs['split'] = split

    print(f"Saved {n_images} images to {h5_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FFHQ dataset for super-resolution training'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing FFHQ images')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--hr-size', type=int, default=256,
                        help='Target HR image size (default: 256)')
    parser.add_argument('--lr-size', type=int, default=64,
                        help='Target LR image size (default: 64)')
    parser.add_argument('--degradation', type=str, default='bicubic',
                        choices=['bicubic', 'bilinear', 'realistic'],
                        help='Degradation method (default: bicubic)')
    parser.add_argument('--save-hdf5', action='store_true',
                        help='Also save data in HDF5 format')
    parser.add_argument('--train-ratio', type=float, default=0.857,
                        help='Training set ratio (default: 0.857 for ~60k)')
    parser.add_argument('--val-ratio', type=float, default=0.071,
                        help='Validation set ratio (default: 0.071 for ~5k)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without processing')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Get all image files
    print(f"Scanning {input_dir} for images...")
    files = get_image_files(input_dir)
    print(f"Found {len(files)} images")

    if not files:
        print("No images found. Exiting.")
        return

    # Limit images if specified
    if args.max_images:
        files = files[:args.max_images]
        print(f"Limited to {len(files)} images")

    # Split dataset
    train_files, val_files, test_files = split_dataset(
        files, args.train_ratio, args.val_ratio, args.seed
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")
    print(f"  Test:  {len(test_files)} images")

    if args.dry_run:
        print("\n[Dry run] No files were processed.")
        return

    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'

    # Process each split
    print(f"\nProcessing with {args.degradation} degradation...")
    print(f"HR size: {args.hr_size}x{args.hr_size}")
    print(f"LR size: {args.lr_size}x{args.lr_size}")

    n_train = process_and_save_images(
        train_files, train_dir, args.hr_size, args.lr_size,
        args.degradation, "Processing train"
    )

    n_val = process_and_save_images(
        val_files, val_dir, args.hr_size, args.lr_size,
        args.degradation, "Processing val"
    )

    n_test = process_and_save_images(
        test_files, test_dir, args.hr_size, args.lr_size,
        args.degradation, "Processing test"
    )

    # Save to HDF5 if requested
    if args.save_hdf5:
        print("\nConverting to HDF5 format...")
        save_to_hdf5(train_dir, 'train', args.hr_size, args.lr_size)
        save_to_hdf5(val_dir, 'val', args.hr_size, args.lr_size)
        save_to_hdf5(test_dir, 'test', args.hr_size, args.lr_size)

    # Save metadata
    metadata = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'hr_size': args.hr_size,
        'lr_size': args.lr_size,
        'scale_factor': args.hr_size // args.lr_size,
        'degradation': args.degradation,
        'seed': args.seed,
        'splits': {
            'train': n_train,
            'val': n_val,
            'test': n_test
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print("Dataset preparation complete!")
    print(f"{'='*50}")
    print(f"Total processed: {n_train + n_val + n_test} images")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")


if __name__ == '__main__':
    main()
