"""
Split Data Script
=================
Splits images from a source directory into train/val/test sets.
Does NOT modify images - just copies them to split folders.
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def split_data(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.857,
    val_ratio: float = 0.071,
    test_ratio: float = 0.072,
    seed: int = 42,
    copy: bool = True,
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
):
    """
    Split images from source directory into train/val/test sets.

    Args:
        source_dir: Directory containing source images
        output_dir: Output directory for split data
        train_ratio: Ratio for training set (default: 85.7%)
        val_ratio: Ratio for validation set (default: 7.1%)
        test_ratio: Ratio for test set (default: 7.2%)
        seed: Random seed for reproducibility
        copy: If True, copy files. If False, move files.
        extensions: Tuple of valid image extensions
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio:.3f}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    print(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

    # Set random seed
    random.seed(seed)

    # Get all image files
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    image_files = []
    for ext in extensions:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))  # Remove duplicates and sort

    if not image_files:
        raise ValueError(f"No images found in {source_dir} with extensions {extensions}")

    print(f"Found {len(image_files)} images in {source_dir}")

    # Shuffle files
    random.shuffle(image_files)

    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remainder goes to test

    print(f"Split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    # Copy/move function
    operation = shutil.copy2 if copy else shutil.move
    op_name = "Copying" if copy else "Moving"

    # Process each split
    splits = [
        ("train", train_dir, train_files),
        ("val", val_dir, val_files),
        ("test", test_dir, test_files),
    ]

    for split_name, split_dir, files in splits:
        print(f"\n{op_name} {len(files)} files to {split_name}/...")
        for f in tqdm(files, desc=split_name):
            dst = split_dir / f.name
            operation(str(f), str(dst))

    # Summary
    print("\n" + "="*50)
    print("SPLIT COMPLETE")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"  train/: {len(list(train_dir.iterdir()))} images")
    print(f"  val/:   {len(list(val_dir.iterdir()))} images")
    print(f"  test/:  {len(list(test_dir.iterdir()))} images")

    return {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split images into train/val/test sets without modification"
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source directory containing images'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for split data'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.857,
        help='Training set ratio (default: 0.857 = 85.7%%)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.071,
        help='Validation set ratio (default: 0.071 = 7.1%%)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.072,
        help='Test set ratio (default: 0.072 = 7.2%%)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying (default: copy)'
    )

    args = parser.parse_args()

    split_data(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy=not args.move
    )


if __name__ == '__main__':
    main()
