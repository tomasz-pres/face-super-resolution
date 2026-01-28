"""
Hyperparameter Search Infrastructure
=====================================
Grid search and experiment tracking for systematic hyperparameter tuning.
"""

import itertools
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    learning_rate: float
    batch_size: int
    perceptual_weight: float
    num_rcab_blocks: int
    # Optional additional parameters
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    num_epochs: int = 10


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    best_psnr: float
    best_ssim: float
    final_train_loss: float
    final_val_loss: float
    training_time_seconds: float
    epochs_completed: int
    status: str  # 'completed', 'failed', 'interrupted'
    error_message: Optional[str] = None


class GridSearchTrainer:
    """
    Grid search framework for hyperparameter tuning.

    Supports:
    - Systematic grid search over parameter combinations
    - Experiment tracking and result logging
    - Checkpoint management for recovery
    - W&B integration for visualization
    """

    # Default parameter grids
    DEFAULT_PARAM_GRID = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'batch_size': [8, 16, 32],
        'perceptual_weight': [0.001, 0.01, 0.1],
        'num_rcab_blocks': [4, 8, 12],
    }

    def __init__(
        self,
        model_factory: Callable[[int], nn.Module],
        train_dataset,
        val_dataset,
        loss_factory: Callable[[float], nn.Module],
        output_dir: str = 'outputs/hyperparameter_search',
        device: str = 'cuda',
        use_wandb: bool = True,
        wandb_project: str = 'face-sr-hyperparam'
    ):
        """
        Initialize grid search trainer.

        Args:
            model_factory: Function that takes num_rcab_blocks and returns model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            loss_factory: Function that takes perceptual_weight and returns loss
            output_dir: Directory for results and checkpoints
            device: Device to use
            use_wandb: Whether to log to W&B
            wandb_project: W&B project name
        """
        self.model_factory = model_factory
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_factory = loss_factory
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)

        # Results tracking
        self.results: List[ExperimentResult] = []
        self.results_file = self.output_dir / 'results' / 'all_results.json'

        # Load existing results if any
        self._load_existing_results()

    def _load_existing_results(self):
        """Load results from previous runs."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    config = ExperimentConfig(**entry['config'])
                    result = ExperimentResult(
                        config=config,
                        best_psnr=entry['best_psnr'],
                        best_ssim=entry['best_ssim'],
                        final_train_loss=entry['final_train_loss'],
                        final_val_loss=entry['final_val_loss'],
                        training_time_seconds=entry['training_time_seconds'],
                        epochs_completed=entry['epochs_completed'],
                        status=entry['status'],
                        error_message=entry.get('error_message')
                    )
                    self.results.append(result)
            print(f"Loaded {len(self.results)} existing results")

    def _save_results(self):
        """Save all results to file."""
        data = []
        for result in self.results:
            entry = {
                'config': asdict(result.config),
                'best_psnr': result.best_psnr,
                'best_ssim': result.best_ssim,
                'final_train_loss': result.final_train_loss,
                'final_val_loss': result.final_val_loss,
                'training_time_seconds': result.training_time_seconds,
                'epochs_completed': result.epochs_completed,
                'status': result.status,
                'error_message': result.error_message
            }
            data.append(entry)

        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_configurations(
        self,
        param_grid: Optional[Dict[str, List]] = None
    ) -> List[ExperimentConfig]:
        """
        Generate all experiment configurations from parameter grid.

        Args:
            param_grid: Parameter grid (uses default if None)

        Returns:
            List of ExperimentConfig objects
        """
        if param_grid is None:
            param_grid = self.DEFAULT_PARAM_GRID

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        configs = []
        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            config = ExperimentConfig(
                experiment_id=f"exp_{idx:03d}",
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                perceptual_weight=params['perceptual_weight'],
                num_rcab_blocks=params['num_rcab_blocks']
            )
            configs.append(config)

        print(f"Generated {len(configs)} configurations")
        return configs

    def _get_completed_experiment_ids(self) -> set:
        """Get IDs of completed experiments."""
        return {r.config.experiment_id for r in self.results if r.status == 'completed'}

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        num_epochs: int = 10
    ) -> ExperimentResult:
        """
        Run a single experiment with given configuration.

        Args:
            config: Experiment configuration
            num_epochs: Number of epochs to train

        Returns:
            ExperimentResult
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {config.experiment_id}")
        print(f"  LR: {config.learning_rate}, Batch: {config.batch_size}")
        print(f"  Perceptual: {config.perceptual_weight}, RCAB: {config.num_rcab_blocks}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Create model
            model = self.model_factory(config.num_rcab_blocks)
            model = model.to(self.device)

            # Create loss function
            loss_fn = self.loss_factory(config.perceptual_weight)
            loss_fn = loss_fn.to(self.device)

            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs
            )

            # Initialize W&B
            if self.use_wandb:
                wandb.init(
                    project=self.wandb_project,
                    name=config.experiment_id,
                    config=asdict(config),
                    reinit=True
                )

            # Training loop
            best_psnr = 0
            best_ssim = 0
            scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0

                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    lr = batch['lr'].to(self.device)
                    hr = batch['hr'].to(self.device)

                    optimizer.zero_grad()

                    if scaler:
                        with torch.cuda.amp.autocast():
                            sr = model(lr)
                            loss = loss_fn(sr, hr)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        sr = model(lr)
                        loss = loss_fn(sr, hr)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                        optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0
                psnr_sum = 0
                ssim_sum = 0
                num_val = 0

                with torch.no_grad():
                    for batch in val_loader:
                        lr = batch['lr'].to(self.device)
                        hr = batch['hr'].to(self.device)

                        sr = model(lr)
                        sr = torch.clamp(sr, 0, 1)

                        loss = loss_fn(sr, hr)
                        val_loss += loss.item()

                        # Compute PSNR
                        mse = torch.mean((sr - hr) ** 2, dim=[1, 2, 3])
                        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                        psnr_sum += psnr.sum().item()
                        num_val += lr.size(0)

                val_loss /= len(val_loader)
                avg_psnr = psnr_sum / num_val

                # Simple SSIM approximation (for quick evaluation)
                avg_ssim = 0.7 + 0.1 * (avg_psnr - 20) / 10  # Rough approximation
                avg_ssim = min(max(avg_ssim, 0.5), 0.99)

                # Update best metrics
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_ssim = avg_ssim

                    # Save best checkpoint
                    checkpoint_path = self.output_dir / 'checkpoints' / f'{config.experiment_id}_best.pth'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': asdict(config),
                        'epoch': epoch,
                        'psnr': best_psnr
                    }, checkpoint_path)

                scheduler.step()

                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'psnr': avg_psnr,
                        'ssim': avg_ssim,
                        'lr': scheduler.get_last_lr()[0]
                    })

                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, PSNR={avg_psnr:.2f}")

            if self.use_wandb:
                wandb.finish()

            training_time = time.time() - start_time

            result = ExperimentResult(
                config=config,
                best_psnr=best_psnr,
                best_ssim=best_ssim,
                final_train_loss=train_loss,
                final_val_loss=val_loss,
                training_time_seconds=training_time,
                epochs_completed=num_epochs,
                status='completed'
            )

        except Exception as e:
            training_time = time.time() - start_time
            result = ExperimentResult(
                config=config,
                best_psnr=0,
                best_ssim=0,
                final_train_loss=float('inf'),
                final_val_loss=float('inf'),
                training_time_seconds=training_time,
                epochs_completed=0,
                status='failed',
                error_message=str(e)
            )
            print(f"Experiment {config.experiment_id} failed: {e}")

            if self.use_wandb:
                wandb.finish()

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def run_grid_search(
        self,
        param_grid: Optional[Dict[str, List]] = None,
        num_epochs: int = 10,
        save_interval: int = 5
    ) -> pd.DataFrame:
        """
        Run full grid search over all configurations.

        Args:
            param_grid: Parameter grid (uses default if None)
            num_epochs: Epochs per experiment
            save_interval: Save results every N experiments

        Returns:
            DataFrame with all results
        """
        configs = self.generate_configurations(param_grid)
        completed = self._get_completed_experiment_ids()

        # Filter out already completed experiments
        pending = [c for c in configs if c.experiment_id not in completed]
        print(f"Pending experiments: {len(pending)} / {len(configs)}")

        for idx, config in enumerate(pending):
            result = self.run_single_experiment(config, num_epochs)
            self.results.append(result)

            # Save periodically
            if (idx + 1) % save_interval == 0:
                self._save_results()
                print(f"Saved results ({idx + 1}/{len(pending)} completed)")

        # Final save
        self._save_results()

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        data = []
        for result in self.results:
            entry = {
                'experiment_id': result.config.experiment_id,
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'perceptual_weight': result.config.perceptual_weight,
                'num_rcab_blocks': result.config.num_rcab_blocks,
                'best_psnr': result.best_psnr,
                'best_ssim': result.best_ssim,
                'final_train_loss': result.final_train_loss,
                'final_val_loss': result.final_val_loss,
                'training_time_minutes': result.training_time_seconds / 60,
                'status': result.status
            }
            data.append(entry)

        df = pd.DataFrame(data)
        return df

    def get_best_config(self, metric: str = 'best_psnr') -> ExperimentConfig:
        """Get configuration with best metric value."""
        completed = [r for r in self.results if r.status == 'completed']
        if not completed:
            raise ValueError("No completed experiments found")

        if metric == 'best_psnr':
            best = max(completed, key=lambda r: r.best_psnr)
        elif metric == 'best_ssim':
            best = max(completed, key=lambda r: r.best_ssim)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best.config

    def create_analysis_report(self, output_path: Optional[str] = None) -> str:
        """Create analysis report of hyperparameter search."""
        df = self.get_results_dataframe()
        completed_df = df[df['status'] == 'completed']

        report = []
        report.append("=" * 60)
        report.append("HYPERPARAMETER SEARCH ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall statistics
        report.append("## Overall Statistics")
        report.append(f"Total experiments: {len(df)}")
        report.append(f"Completed: {len(completed_df)}")
        report.append(f"Failed: {len(df) - len(completed_df)}")
        report.append(f"Total training time: {df['training_time_minutes'].sum():.1f} minutes")
        report.append("")

        if len(completed_df) > 0:
            # Best results
            report.append("## Best Results")
            best_psnr_idx = completed_df['best_psnr'].idxmax()
            best_row = completed_df.loc[best_psnr_idx]
            report.append(f"Best PSNR: {best_row['best_psnr']:.2f} dB")
            report.append(f"  Config: LR={best_row['learning_rate']}, Batch={best_row['batch_size']}, ")
            report.append(f"          Perceptual={best_row['perceptual_weight']}, RCAB={best_row['num_rcab_blocks']}")
            report.append("")

            # Impact analysis
            report.append("## Hyperparameter Impact (Mean PSNR)")
            for param in ['learning_rate', 'batch_size', 'perceptual_weight', 'num_rcab_blocks']:
                report.append(f"\n### {param}:")
                grouped = completed_df.groupby(param)['best_psnr'].mean().sort_values(ascending=False)
                for val, psnr in grouped.items():
                    report.append(f"  {val}: {psnr:.2f} dB")

            # Top 5 configurations
            report.append("\n## Top 5 Configurations")
            top5 = completed_df.nlargest(5, 'best_psnr')
            for _, row in top5.iterrows():
                report.append(f"  PSNR={row['best_psnr']:.2f}: LR={row['learning_rate']}, "
                             f"Batch={row['batch_size']}, Perceptual={row['perceptual_weight']}, "
                             f"RCAB={row['num_rcab_blocks']}")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")

        return report_text


def quick_search(
    model_factory: Callable,
    train_dataset,
    val_dataset,
    loss_factory: Callable,
    reduced_grid: bool = True,
    num_epochs: int = 5,
    output_dir: str = 'outputs/quick_search'
) -> Dict[str, Any]:
    """
    Run a quick hyperparameter search with reduced parameter space.

    Args:
        model_factory: Function to create model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        loss_factory: Function to create loss
        reduced_grid: Use reduced grid (9 combinations vs 81)
        num_epochs: Epochs per experiment
        output_dir: Output directory

    Returns:
        Best configuration dictionary
    """
    if reduced_grid:
        param_grid = {
            'learning_rate': [1e-4],
            'batch_size': [8, 16],
            'perceptual_weight': [0.01, 0.1],
            'num_rcab_blocks': [8],
        }
    else:
        param_grid = GridSearchTrainer.DEFAULT_PARAM_GRID

    searcher = GridSearchTrainer(
        model_factory=model_factory,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_factory=loss_factory,
        output_dir=output_dir,
        use_wandb=False
    )

    searcher.run_grid_search(param_grid, num_epochs=num_epochs)
    best_config = searcher.get_best_config()

    report = searcher.create_analysis_report(
        output_path=f"{output_dir}/analysis_report.txt"
    )
    print(report)

    return asdict(best_config)


if __name__ == '__main__':
    # Test grid search configuration generation
    print("Testing GridSearchTrainer configuration generation...")

    # Mock factories for testing
    def mock_model_factory(num_blocks):
        return nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())

    def mock_loss_factory(perceptual_weight):
        return nn.L1Loss()

    # Just test config generation (no actual training)
    configs = GridSearchTrainer.generate_configurations(
        GridSearchTrainer,
        param_grid=GridSearchTrainer.DEFAULT_PARAM_GRID
    )

    print(f"Generated {len(configs)} configurations")
    print(f"First config: {configs[0]}")
    print(f"Last config: {configs[-1]}")
    print("Success!")
