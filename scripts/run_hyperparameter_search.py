#!/usr/bin/env python
"""
Hyperparameter Search Script
==============================
Run systematic hyperparameter search for face super-resolution models.

Usage:
    python scripts/run_hyperparameter_search.py --quick          # Quick 4-config search
    python scripts/run_hyperparameter_search.py --full           # Full 81-config search
    python scripts/run_hyperparameter_search.py --resume         # Resume interrupted search
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.data import get_dataloader, FFHQDataset
from src.models import create_face_enhance_net
from src.losses import CombinedLoss
from src.training.hyperparameter_search import GridSearchTrainer, quick_search


def create_model_factory():
    """Create model factory function."""
    def factory(num_rcab_blocks: int):
        return create_face_enhance_net(
            num_rcab=num_rcab_blocks,
            num_features=64,
            scale_factor=4
        )
    return factory


def create_loss_factory(device: str = 'cuda'):
    """Create loss factory function."""
    def factory(perceptual_weight: float):
        return CombinedLoss(
            l1_weight=1.0,
            perceptual_weight=perceptual_weight,
            ssim_weight=0.1,
            device=device
        )
    return factory


def run_quick_search(args):
    """Run quick hyperparameter search (4 configurations)."""
    print("Running QUICK hyperparameter search...")
    print("Configurations: 4 (2 batch sizes x 2 perceptual weights)")
    print("Epochs per config: 5")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load datasets
    print(f"Loading data from {args.data_root}...")
    train_dataset = FFHQDataset(args.data_root, mode='train')
    val_dataset = FFHQDataset(args.data_root, mode='val')
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Run quick search
    best_config = quick_search(
        model_factory=create_model_factory(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_factory=create_loss_factory(device),
        reduced_grid=True,
        num_epochs=5,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION FOUND:")
    print("=" * 60)
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    return best_config


def run_full_search(args):
    """Run full hyperparameter search (81 configurations)."""
    print("Running FULL hyperparameter search...")
    print("Configurations: 81 (3 LR x 3 batch x 3 perceptual x 3 RCAB)")
    print("Epochs per config: 10")
    print("Estimated time: ~8-12 hours on GPU")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load datasets
    print(f"Loading data from {args.data_root}...")
    train_dataset = FFHQDataset(args.data_root, mode='train')
    val_dataset = FFHQDataset(args.data_root, mode='val')
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create searcher
    searcher = GridSearchTrainer(
        model_factory=create_model_factory(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_factory=create_loss_factory(device),
        output_dir=args.output_dir,
        device=device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project
    )

    # Run full grid search
    results_df = searcher.run_grid_search(
        num_epochs=args.epochs,
        save_interval=5
    )

    # Generate report
    report = searcher.create_analysis_report(
        output_path=f"{args.output_dir}/analysis_report.txt"
    )
    print(report)

    # Save results as CSV
    results_df.to_csv(f"{args.output_dir}/results.csv", index=False)
    print(f"\nResults saved to {args.output_dir}/results.csv")

    # Get best config
    best_config = searcher.get_best_config()
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION:")
    print("=" * 60)
    print(f"  Learning Rate: {best_config.learning_rate}")
    print(f"  Batch Size: {best_config.batch_size}")
    print(f"  Perceptual Weight: {best_config.perceptual_weight}")
    print(f"  RCAB Blocks: {best_config.num_rcab_blocks}")

    return best_config


def run_custom_search(args):
    """Run custom hyperparameter search with user-specified grid."""
    print("Running CUSTOM hyperparameter search...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parse custom grid from args
    param_grid = {
        'learning_rate': [float(x) for x in args.learning_rates.split(',')],
        'batch_size': [int(x) for x in args.batch_sizes.split(',')],
        'perceptual_weight': [float(x) for x in args.perceptual_weights.split(',')],
        'num_rcab_blocks': [int(x) for x in args.rcab_blocks.split(',')]
    }

    num_configs = 1
    for v in param_grid.values():
        num_configs *= len(v)

    print(f"Configurations: {num_configs}")
    print(f"Parameter grid: {param_grid}")
    print()

    # Load datasets
    print(f"Loading data from {args.data_root}...")
    train_dataset = FFHQDataset(args.data_root, mode='train')
    val_dataset = FFHQDataset(args.data_root, mode='val')
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create searcher
    searcher = GridSearchTrainer(
        model_factory=create_model_factory(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_factory=create_loss_factory(device),
        output_dir=args.output_dir,
        device=device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project
    )

    # Run search
    results_df = searcher.run_grid_search(
        param_grid=param_grid,
        num_epochs=args.epochs,
        save_interval=args.save_interval
    )

    # Generate report
    report = searcher.create_analysis_report(
        output_path=f"{args.output_dir}/analysis_report.txt"
    )
    print(report)

    return searcher.get_best_config()


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Face SR')

    # Mode selection
    parser.add_argument('--quick', action='store_true',
                        help='Run quick search (4 configurations, ~30 min)')
    parser.add_argument('--full', action='store_true',
                        help='Run full search (81 configurations, ~8-12 hours)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted search')

    # Data
    parser.add_argument('--data-root', type=str, default='data/processed',
                        help='Path to processed data')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/hyperparameter_search',
                        help='Output directory')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs per configuration')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save results every N experiments')

    # Custom grid (for advanced users)
    parser.add_argument('--learning-rates', type=str, default='1e-3,1e-4,1e-5',
                        help='Comma-separated learning rates')
    parser.add_argument('--batch-sizes', type=str, default='8,16,32',
                        help='Comma-separated batch sizes')
    parser.add_argument('--perceptual-weights', type=str, default='0.001,0.01,0.1',
                        help='Comma-separated perceptual loss weights')
    parser.add_argument('--rcab-blocks', type=str, default='4,8,12',
                        help='Comma-separated RCAB block counts')

    # W&B
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='face-sr-hyperparam',
                        help='W&B project name')

    args = parser.parse_args()

    # Determine mode
    if args.quick:
        run_quick_search(args)
    elif args.full or args.resume:
        run_full_search(args)
    else:
        # Default to custom search with provided parameters
        run_custom_search(args)

    print("\nHyperparameter search complete!")


if __name__ == '__main__':
    main()
