"""
Training Callbacks and Utilities
=================================
Gradient tracking, activation statistics, and other monitoring utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import numpy as np


class GradientMonitor:
    """
    Monitor gradient statistics during training.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to monitor
        """
        self.model = model
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []

    def register_hooks(self) -> None:
        """Register backward hooks to capture gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._save_gradient(n, grad)
                )
                self.hooks.append(hook)

    def _save_gradient(self, name: str, grad: torch.Tensor) -> None:
        """Save gradient norm for a parameter."""
        norm = grad.norm().item()
        self.gradient_norms[name].append(norm)

    def get_gradient_norms(self) -> Dict[str, float]:
        """Get current gradient norms."""
        return {
            name: norms[-1] if norms else 0.0
            for name, norms in self.gradient_norms.items()
        }

    def get_total_norm(self) -> float:
        """Get total gradient norm."""
        norms = self.get_gradient_norms()
        if not norms:
            return 0.0
        return np.sqrt(sum(n ** 2 for n in norms.values()))

    def get_layer_summary(self) -> Dict[str, Dict[str, float]]:
        """Get gradient statistics per layer type."""
        summary = defaultdict(lambda: {'mean': 0.0, 'max': 0.0, 'count': 0})

        for name, norms in self.gradient_norms.items():
            if not norms:
                continue

            # Extract layer type
            parts = name.split('.')
            layer_type = parts[0] if parts else 'unknown'

            current = norms[-1]
            summary[layer_type]['mean'] += current
            summary[layer_type]['max'] = max(summary[layer_type]['max'], current)
            summary[layer_type]['count'] += 1

        # Compute averages
        for layer_type, stats in summary.items():
            if stats['count'] > 0:
                stats['mean'] /= stats['count']

        return dict(summary)

    def clear(self) -> None:
        """Clear recorded gradients."""
        self.gradient_norms.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ActivationMonitor:
    """
    Monitor activation statistics during training.
    """

    def __init__(self, model: nn.Module, layer_types: tuple = (nn.Conv2d, nn.Linear)):
        """
        Args:
            model: Model to monitor
            layer_types: Types of layers to monitor
        """
        self.model = model
        self.layer_types = layer_types
        self.activation_stats: Dict[str, Dict[str, float]] = {}
        self.hooks = []

    def register_hooks(self) -> None:
        """Register forward hooks to capture activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._save_activation(n, out)
                )
                self.hooks.append(hook)

    def _save_activation(self, name: str, activation: torch.Tensor) -> None:
        """Save activation statistics."""
        with torch.no_grad():
            self.activation_stats[name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'sparsity': (activation == 0).float().mean().item(),
            }

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current activation statistics."""
        return self.activation_stats.copy()

    def get_dead_neurons(self, threshold: float = 0.99) -> Dict[str, float]:
        """Get percentage of dead neurons (always zero activations)."""
        dead = {}
        for name, stats in self.activation_stats.items():
            if stats['sparsity'] > threshold:
                dead[name] = stats['sparsity']
        return dead

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class WeightMonitor:
    """
    Monitor weight statistics and updates during training.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to monitor
        """
        self.model = model
        self.prev_weights: Dict[str, torch.Tensor] = {}
        self.update_ratios: Dict[str, List[float]] = defaultdict(list)

    def snapshot_weights(self) -> None:
        """Take snapshot of current weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_weights[name] = param.data.clone()

    def compute_update_ratios(self) -> Dict[str, float]:
        """Compute weight update ratios (|delta_w| / |w|)."""
        ratios = {}

        for name, param in self.model.named_parameters():
            if name in self.prev_weights and param.requires_grad:
                delta = (param.data - self.prev_weights[name]).norm()
                weight_norm = param.data.norm()

                if weight_norm > 0:
                    ratio = delta.item() / weight_norm.item()
                else:
                    ratio = 0.0

                ratios[name] = ratio
                self.update_ratios[name].append(ratio)

        return ratios

    def get_weight_stats(self) -> Dict[str, Dict[str, float]]:
        """Get weight statistics for all parameters."""
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    stats[name] = {
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'norm': param.data.norm().item(),
                    }
        return stats


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: Any, batch: int) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer: Any, batch: int, metrics: Dict) -> None:
        """Called at the end of each batch."""
        pass


class MetricLogger(TrainingCallback):
    """Log metrics to file."""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        self.metrics_history = []

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        self.metrics_history.append({
            'epoch': epoch,
            **metrics
        })

    def on_train_end(self, trainer: Any) -> None:
        import json
        import os

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'metrics.json')

        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class LRWarmup:
    """Learning rate warmup scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int = 500,
                 initial_lr: float = 1e-7):
        """
        Args:
            optimizer: Optimizer to adjust
            warmup_steps: Number of warmup steps
            initial_lr: Initial learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self) -> float:
        """Perform warmup step."""
        if self.current_step >= self.warmup_steps:
            return self.target_lr

        # Linear warmup
        lr = self.initial_lr + (self.target_lr - self.initial_lr) * (
            self.current_step / self.warmup_steps
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
        return lr

    def is_done(self) -> bool:
        """Check if warmup is complete."""
        return self.current_step >= self.warmup_steps
