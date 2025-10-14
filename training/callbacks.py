"""
Training Callbacks

Callbacks for:
- Early stopping
- Model checkpointing
- Learning rate monitoring
- Memory monitoring
- Causal metrics logging
- Weights & Biases integration
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np
from training.utils import get_memory_usage, clear_memory


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, trainer, step: int):
        """Called at the beginning of each training step."""
        pass

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Called at the end of each training step."""
        pass

    def on_validation_begin(self, trainer):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.

    Stops training if validation metric doesn't improve for patience epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric: str = "causal_stability",
        mode: str = "max"
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
            mode: "min" or "max" (whether lower or higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode

        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False

    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Check if should stop training."""
        if self.metric not in metrics:
            print(f"Warning: Metric '{self.metric}' not found in validation metrics")
            return

        current_value = metrics[self.metric]

        # Check if improved
        if self.mode == 'max':
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
            print(f"Validation {self.metric} improved to {current_value:.4f}")
        else:
            self.counter += 1
            print(f"No improvement in {self.metric} for {self.counter}/{self.patience} validations")

            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} validations without improvement")
                self.should_stop = True
                trainer.should_stop = True


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback.

    Saves the best model based on validation metric.
    """

    def __init__(
        self,
        output_dir: str,
        metric: str = "causal_stability",
        mode: str = "max",
        save_total_limit: int = 3,
        save_every_n_steps: Optional[int] = None
    ):
        """
        Initialize model checkpointing.

        Args:
            output_dir: Directory to save checkpoints
            metric: Metric to monitor for best model
            mode: "min" or "max"
            save_total_limit: Maximum number of checkpoints to keep
            save_every_n_steps: Save checkpoint every N steps (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metric = metric
        self.mode = mode
        self.save_total_limit = save_total_limit
        self.save_every_n_steps = save_every_n_steps

        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.checkpoint_paths = []

    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Save checkpoint if metric improved."""
        if self.metric not in metrics:
            return

        current_value = metrics[self.metric]

        # Check if this is the best model
        is_best = False
        if self.mode == 'max':
            is_best = current_value > self.best_value
        else:
            is_best = current_value < self.best_value

        if is_best:
            self.best_value = current_value
            self._save_checkpoint(trainer, is_best=True, metric_value=current_value)

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Save periodic checkpoint."""
        if self.save_every_n_steps and step % self.save_every_n_steps == 0:
            self._save_checkpoint(trainer, is_best=False, step=step)

    def _save_checkpoint(
        self,
        trainer,
        is_best: bool = False,
        metric_value: Optional[float] = None,
        step: Optional[int] = None
    ):
        """Save a checkpoint."""
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "best_model"
        elif step is not None:
            checkpoint_name = f"checkpoint-{step}"
        else:
            checkpoint_name = f"checkpoint-{trainer.global_step}"

        checkpoint_path = self.output_dir / checkpoint_name

        # Save model
        trainer.model.save_pretrained(checkpoint_path)
        trainer.tokenizer.save_pretrained(checkpoint_path)

        # Save causal projection weights
        if hasattr(trainer.model, 'causal_projection'):
            torch.save(
                trainer.model.causal_projection.state_dict(),
                checkpoint_path / "causal_projection.pt"
            )
            print(f"Causal projection saved to {checkpoint_path / 'causal_projection.pt'}")

        # Save trainer state
        torch.save({
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "best_metric": self.best_value,
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict() if trainer.scheduler else None
        }, checkpoint_path / "trainer_state.pt")

        print(f"Checkpoint saved to {checkpoint_path}")

        if not is_best:
            # Track non-best checkpoints for cleanup
            self.checkpoint_paths.append(checkpoint_path)

            # Remove old checkpoints if exceeding limit
            if len(self.checkpoint_paths) > self.save_total_limit:
                old_checkpoint = self.checkpoint_paths.pop(0)
                if old_checkpoint.exists():
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    print(f"Removed old checkpoint: {old_checkpoint}")


class LearningRateMonitor(Callback):
    """Monitor and log learning rate changes."""

    def __init__(self):
        """Initialize LR monitor."""
        self.learning_rates = []

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Log current learning rate."""
        if trainer.optimizer:
            lr = trainer.optimizer.param_groups[0]['lr']
            self.learning_rates.append(lr)

            # Add to metrics for logging
            metrics['learning_rate'] = lr


class MemoryMonitor(Callback):
    """
    Monitor GPU memory usage during training.

    Helps identify memory leaks and optimization opportunities.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        clear_cache_every_n_steps: int = 100
    ):
        """
        Initialize memory monitor.

        Args:
            log_every_n_steps: Log memory every N steps
            clear_cache_every_n_steps: Clear cache every N steps
        """
        self.log_every_n_steps = log_every_n_steps
        self.clear_cache_every_n_steps = clear_cache_every_n_steps
        self.memory_history = []

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Monitor memory usage."""
        if step % self.log_every_n_steps == 0:
            memory = get_memory_usage()
            self.memory_history.append({
                "step": step,
                **memory
            })

            # Log memory metrics
            metrics['memory_allocated_gb'] = memory['allocated']
            metrics['memory_reserved_gb'] = memory['reserved']

            print(f"Step {step} - Memory: {memory['allocated']:.2f}GB allocated, "
                  f"{memory['reserved']:.2f}GB reserved")

        # Periodically clear cache to prevent memory fragmentation
        if step % self.clear_cache_every_n_steps == 0:
            clear_memory()

    def on_validation_begin(self, trainer):
        """Clear cache before validation."""
        clear_memory()

    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Clear cache after validation."""
        clear_memory()


class CausalMetricsLogger(Callback):
    """
    Log causal-specific metrics during training.

    Tracks:
    - Causal stability
    - Spurious separation
    - Task loss
    - Contrastive loss
    """

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics_history = []

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Log causal metrics."""
        # Extract causal metrics
        causal_metrics = {}

        for key in ['causal_stability', 'spurious_separation', 'task_loss', 'contrastive_loss']:
            if key in metrics:
                causal_metrics[key] = metrics[key]

        if causal_metrics:
            self.metrics_history.append({
                "step": step,
                **causal_metrics
            })

    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Log validation causal metrics."""
        print("\nValidation Causal Metrics:")
        for key in ['causal_stability', 'spurious_separation', 'causal_discrimination']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")


class WandbLogger(Callback):
    """
    Weights & Biases logging callback.

    Logs all metrics to W&B for experiment tracking.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (username/org)
            name: Run name
            config: Configuration dictionary to log
            tags: Tags for the run
            notes: Notes for the run
        """
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            self.enabled = False
            return

        # Initialize W&B run
        self.run = self.wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes
        )

    def on_train_begin(self, trainer):
        """Log training start."""
        if not self.enabled:
            return

        # Log model info
        self.wandb.log({
            "model_name": trainer.model_name,
            "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        })

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Log step metrics."""
        if not self.enabled:
            return

        # Log all metrics
        self.wandb.log({
            "step": step,
            "epoch": trainer.current_epoch,
            **metrics
        })

    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Log validation metrics."""
        if not self.enabled:
            return

        # Prefix validation metrics
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        self.wandb.log({
            "step": trainer.global_step,
            **val_metrics
        })

    def on_train_end(self, trainer):
        """Finish W&B run."""
        if not self.enabled:
            return

        self.wandb.finish()


class ProgressLogger(Callback):
    """
    Simple progress logger for console output.

    Provides clean, informative progress updates during training.
    """

    def __init__(self, log_every_n_steps: int = 10):
        """
        Initialize progress logger.

        Args:
            log_every_n_steps: Log progress every N steps
        """
        self.log_every_n_steps = log_every_n_steps
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, trainer):
        """Start training timer."""
        self.train_start_time = time.time()
        print("\n" + "="*80)
        print("TRAINING STARTED")
        print("="*80)

    def on_epoch_begin(self, trainer, epoch: int):
        """Start epoch timer."""
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{trainer.num_epochs}")
        print("-" * 80)

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]):
        """Log step progress."""
        if step % self.log_every_n_steps == 0:
            # Format metrics
            loss_str = f"Loss: {metrics.get('loss', 0):.4f}"
            lr_str = f"LR: {metrics.get('learning_rate', 0):.2e}"

            # Causal metrics if available
            causal_str = ""
            if 'causal_stability' in metrics:
                causal_str = f" | Causal: {metrics['causal_stability']:.3f}"
            if 'spurious_separation' in metrics:
                causal_str += f" | Spurious: {metrics['spurious_separation']:.3f}"

            print(f"Step {step}: {loss_str} | {lr_str}{causal_str}")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Log epoch summary."""
        epoch_time = time.time() - self.epoch_start_time

        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average loss: {metrics.get('avg_loss', 0):.4f}")

    def on_train_end(self, trainer):
        """Log training summary."""
        train_time = time.time() - self.train_start_time

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print(f"Total time: {train_time/3600:.2f} hours")
        print("="*80)


if __name__ == "__main__":
    # Test callbacks
    print("Testing callbacks...")

    # Test early stopping
    early_stopping = EarlyStopping(patience=3, metric="loss", mode="min")

    # Simulate validation metrics
    class DummyTrainer:
        should_stop = False

    trainer = DummyTrainer()

    # Metrics improving
    early_stopping.on_validation_end(trainer, {"loss": 1.5})
    early_stopping.on_validation_end(trainer, {"loss": 1.2})

    # Metrics not improving
    early_stopping.on_validation_end(trainer, {"loss": 1.3})
    early_stopping.on_validation_end(trainer, {"loss": 1.4})
    early_stopping.on_validation_end(trainer, {"loss": 1.4})
    early_stopping.on_validation_end(trainer, {"loss": 1.5})

    print(f"Should stop: {early_stopping.should_stop}")

    print("\nCallbacks test completed!")
