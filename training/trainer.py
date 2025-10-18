"""
Custom Trainer for Causal Contrastive Learning

Implements training loop with:
- Causal contrastive loss
- Memory-efficient training for RTX 4090
- Gradient accumulation
- Mixed precision training
- Checkpointing and resumption
- Validation with causal metrics
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler

from models.losses import CausalContrastiveLoss
from training.utils import (
    get_memory_usage,
    clear_memory,
    compute_causal_metrics,
    print_memory_usage
)


class CausalTrainer:
    """
    Custom trainer for causal contrastive learning.

    Optimized for RTX 4090 (24GB VRAM).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        config: Optional[Dict] = None,
        callbacks: Optional[List] = None,
        output_dir: str = "checkpoints",
        device: str = "cuda"
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train (with LoRA)
            tokenizer: Tokenizer
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            loss_fn: Loss function (CausalContrastiveLoss)
            config: Training configuration
            callbacks: List of callbacks
            output_dir: Output directory for checkpoints
            device: Device to train on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn or CausalContrastiveLoss()
        self.config = config or {}
        self.callbacks = callbacks or []
        self.output_dir = Path(output_dir)
        self.device = device

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.current_batch_idx = 0  # Track current batch for checkpointing
        self.should_stop = False

        # Resume tracking (set by load_checkpoint)
        self.resumed_epoch = 0
        self.resumed_batch_idx = 0

        # Training config
        self.num_epochs = self.config.get("training", {}).get("num_epochs", 3)
        self.gradient_accumulation_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 16)
        self.max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
        self.logging_steps = self.config.get("training", {}).get("logging_steps", 10)
        self.eval_steps = self.config.get("training", {}).get("eval_steps", 200)
        self.evaluation_strategy = self.config.get("training", {}).get("evaluation_strategy", "steps")

        # CRITICAL SAFETY: Disable step-based validation if strategy is "epoch"
        if self.evaluation_strategy == "epoch":
            self.eval_steps = float('inf')  # Will never trigger mid-epoch validation
            print("✓ Validation strategy: EPOCH-ONLY (no mid-epoch validation)")
        elif self.evaluation_strategy == "steps":
            print(f"✓ Validation strategy: STEPS (every {self.eval_steps} steps)")

        self.fp16 = self.config.get("training", {}).get("fp16", False)
        self.bf16 = self.config.get("training", {}).get("bf16", True)

        # Mixed precision setup
        self.use_amp = self.fp16 or self.bf16
        self.amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.scaler = GradScaler() if self.fp16 else None

        # Learning rate scheduler
        self.scheduler = None
        if self.optimizer:
            self.scheduler = self._create_scheduler()

        # Model name
        self.model_name = self.config.get("model", {}).get("name", "unknown")

        # Move model to device
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.config.get("training", {}).get("warmup_ratio", 0.03))

        scheduler = get_scheduler(
            name=self.config.get("training", {}).get("lr_scheduler_type", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return scheduler

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.num_epochs} epochs")
        print(f"Total training steps: {len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps}")
        print_memory_usage("Initial ")

        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        try:
            # Resume from saved epoch if checkpoint was loaded
            for epoch in range(self.current_epoch, self.num_epochs):
                if self.should_stop:
                    print("Early stopping triggered")
                    break

                self.current_epoch = epoch
                self._train_epoch(epoch)

                # Validation
                if self.val_dataloader:
                    val_metrics = self.validate()

                    # Trigger callbacks
                    for callback in self.callbacks:
                        callback.on_validation_end(self, val_metrics)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise

        finally:
            # Trigger callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)

        print("\nTraining completed!")

    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_epoch_begin(self, epoch)

        epoch_loss = 0.0
        epoch_metrics = {
            "causal_stability": 0.0,
            "spurious_separation": 0.0,
            "task_loss": 0.0,
            "contrastive_loss": 0.0
        }
        num_batches = 0

        # Progress bar (adjust initial position if resuming mid-epoch)
        if epoch == self.resumed_epoch and self.resumed_batch_idx > 0:
            # Set initial position to show correct progress when resuming
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                initial=self.resumed_batch_idx,
                total=len(self.train_dataloader)
            )
        else:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Update current batch index for checkpointing
            self.current_batch_idx = batch_idx

            # Skip batches if resuming from checkpoint within the same epoch
            if epoch == self.resumed_epoch and self.resumed_batch_idx > 0:
                expected_batch_idx = self.resumed_batch_idx
                if batch_idx < expected_batch_idx:
                    # CRITICAL: Update progress bar for skipped batches
                    pbar.update(1)
                    # Don't trigger callbacks or accumulate metrics for skipped batches
                    continue
                elif batch_idx == expected_batch_idx:
                    print(f"\n{'='*60}")
                    print(f"RESUMING: Batch {batch_idx} | Step {self.global_step}")
                    print(f"{'='*60}\n")
                    self.resumed_batch_idx = 0

            # Trigger callbacks (only for non-skipped batches)
            for callback in self.callbacks:
                callback.on_step_begin(self, self.global_step)

            # Train step
            loss, metrics = self._train_step(batch, batch_idx)

            # Accumulate metrics
            epoch_loss += loss
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "causal": f"{metrics.get('causal_stability', 0):.3f}",
                "spurious": f"{metrics.get('spurious_separation', 0):.3f}"
            })

            # Logging
            if self.global_step % self.logging_steps == 0:
                log_metrics = {
                    "loss": loss,
                    **metrics,
                    "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                }

                # Trigger callbacks
                for callback in self.callbacks:
                    callback.on_step_end(self, self.global_step, log_metrics)

            # Validation (only if evaluation_strategy is "steps", not "epoch")
            if (self.val_dataloader and
                self.evaluation_strategy == "steps" and
                self.global_step > 0 and
                self.global_step % self.eval_steps == 0):
                val_metrics = self.validate()

                # Trigger callbacks
                for callback in self.callbacks:
                    callback.on_validation_end(self, val_metrics)

                # Back to training mode
                self.model.train()

            # Check if should stop
            if self.should_stop:
                break

        # Compute epoch averages
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches if num_batches > 0 else 1

        epoch_metrics["avg_loss"] = avg_loss

        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_epoch_end(self, epoch, epoch_metrics)

    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[float, Dict[str, float]]:
        """
        Single training step with gradient accumulation.

        Args:
            batch: Batch dictionary with triplets
            batch_idx: Batch index

        Returns:
            Tuple of (loss, metrics)
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with mixed precision
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
            # Get representations and logits for all three inputs
            # Benign
            outputs_benign = self.model(
                input_ids=batch["benign_input_ids"],
                attention_mask=batch["benign_attention_mask"],
                output_hidden_states=True
            )
            hidden_states_benign = outputs_benign.hidden_states[-1]
            # Pool with attention mask
            mask_expanded_benign = batch["benign_attention_mask"].unsqueeze(-1).expand(hidden_states_benign.size()).float()
            pooled_benign = (hidden_states_benign * mask_expanded_benign).sum(dim=1) / mask_expanded_benign.sum(dim=1).clamp(min=1e-9)
            # Apply projection
            representation_benign = self.model.causal_projection(pooled_benign)
            benign_outputs = {
                "logits": outputs_benign.logits,
                "representation": representation_benign
            }

            # Benign counterfactual
            outputs_benign_cf = self.model(
                input_ids=batch["benign_cf_input_ids"],
                attention_mask=batch["benign_cf_attention_mask"],
                output_hidden_states=True
            )
            hidden_states_benign_cf = outputs_benign_cf.hidden_states[-1]
            # Pool with attention mask
            mask_expanded_benign_cf = batch["benign_cf_attention_mask"].unsqueeze(-1).expand(hidden_states_benign_cf.size()).float()
            pooled_benign_cf = (hidden_states_benign_cf * mask_expanded_benign_cf).sum(dim=1) / mask_expanded_benign_cf.sum(dim=1).clamp(min=1e-9)
            # Apply projection
            representation_benign_cf = self.model.causal_projection(pooled_benign_cf)
            benign_cf_outputs = {
                "logits": outputs_benign_cf.logits,
                "representation": representation_benign_cf
            }

            # Injection
            outputs_injection = self.model(
                input_ids=batch["injection_input_ids"],
                attention_mask=batch["injection_attention_mask"],
                output_hidden_states=True
            )
            hidden_states_injection = outputs_injection.hidden_states[-1]
            # Pool with attention mask
            mask_expanded_injection = batch["injection_attention_mask"].unsqueeze(-1).expand(hidden_states_injection.size()).float()
            pooled_injection = (hidden_states_injection * mask_expanded_injection).sum(dim=1) / mask_expanded_injection.sum(dim=1).clamp(min=1e-9)
            # Apply projection
            representation_injection = self.model.causal_projection(pooled_injection)
            injection_outputs = {
                "logits": outputs_injection.logits,
                "representation": representation_injection
            }

            # Compute loss
            loss_dict = self.loss_fn(
                repr_benign=benign_outputs["representation"],
                repr_benign_counterfactual=benign_cf_outputs["representation"],
                repr_injection=injection_outputs["representation"],
                logits_benign=benign_outputs["logits"],
                labels_benign=batch["labels"]
            )

            loss = loss_dict["loss"]

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.fp16 and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                if self.fp16 and self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # Optimizer step
            if self.fp16 and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Learning rate scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # Increment global step
            self.global_step += 1

        # Extract metrics (unscale loss)
        metrics = {
            "loss": (loss.item() * self.gradient_accumulation_steps),
            "causal_stability": loss_dict["causal_stability"].item(),
            "spurious_separation": loss_dict["spurious_separation"].item(),
            "task_loss": loss_dict["task_loss"].item(),
            "contrastive_loss": loss_dict["contrastive_loss"].item()
        }

        return metrics["loss"], metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        if not self.val_dataloader:
            return {}

        print("\nRunning validation...")
        self.model.eval()

        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_validation_begin(self)

        # Clear memory before validation
        clear_memory()

        total_loss = 0.0
        all_repr_benign = []
        all_repr_benign_cf = []
        all_repr_injection = []

        metrics_accumulator = {
            "causal_stability": 0.0,
            "spurious_separation": 0.0,
            "task_loss": 0.0,
            "contrastive_loss": 0.0
        }

        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                # Get representations - Benign
                outputs_benign = self.model(
                    input_ids=batch["benign_input_ids"],
                    attention_mask=batch["benign_attention_mask"],
                    output_hidden_states=True
                )
                hidden_states_benign = outputs_benign.hidden_states[-1]
                mask_expanded_benign = batch["benign_attention_mask"].unsqueeze(-1).expand(hidden_states_benign.size()).float()
                pooled_benign = (hidden_states_benign * mask_expanded_benign).sum(dim=1) / mask_expanded_benign.sum(dim=1).clamp(min=1e-9)
                representation_benign = self.model.causal_projection(pooled_benign)
                benign_outputs = {
                    "logits": outputs_benign.logits,
                    "representation": representation_benign
                }

                # Benign counterfactual
                outputs_benign_cf = self.model(
                    input_ids=batch["benign_cf_input_ids"],
                    attention_mask=batch["benign_cf_attention_mask"],
                    output_hidden_states=True
                )
                hidden_states_benign_cf = outputs_benign_cf.hidden_states[-1]
                mask_expanded_benign_cf = batch["benign_cf_attention_mask"].unsqueeze(-1).expand(hidden_states_benign_cf.size()).float()
                pooled_benign_cf = (hidden_states_benign_cf * mask_expanded_benign_cf).sum(dim=1) / mask_expanded_benign_cf.sum(dim=1).clamp(min=1e-9)
                representation_benign_cf = self.model.causal_projection(pooled_benign_cf)
                benign_cf_outputs = {
                    "logits": outputs_benign_cf.logits,
                    "representation": representation_benign_cf
                }

                # Injection
                outputs_injection = self.model(
                    input_ids=batch["injection_input_ids"],
                    attention_mask=batch["injection_attention_mask"],
                    output_hidden_states=True
                )
                hidden_states_injection = outputs_injection.hidden_states[-1]
                mask_expanded_injection = batch["injection_attention_mask"].unsqueeze(-1).expand(hidden_states_injection.size()).float()
                pooled_injection = (hidden_states_injection * mask_expanded_injection).sum(dim=1) / mask_expanded_injection.sum(dim=1).clamp(min=1e-9)
                representation_injection = self.model.causal_projection(pooled_injection)
                injection_outputs = {
                    "logits": outputs_injection.logits,
                    "representation": representation_injection
                }

                # Compute loss
                loss_dict = self.loss_fn(
                    repr_benign=benign_outputs["representation"],
                    repr_benign_counterfactual=benign_cf_outputs["representation"],
                    repr_injection=injection_outputs["representation"],
                    logits_benign=benign_outputs["logits"],
                    labels_benign=batch["labels"]
                )

            # Accumulate
            total_loss += loss_dict["loss"].item()
            for key in metrics_accumulator:
                if key in loss_dict:
                    metrics_accumulator[key] += loss_dict[key].item()

            # Store representations for global metrics
            all_repr_benign.append(benign_outputs["representation"].cpu())
            all_repr_benign_cf.append(benign_cf_outputs["representation"].cpu())
            all_repr_injection.append(injection_outputs["representation"].cpu())

            num_batches += 1

        # Compute averages
        val_metrics = {
            "val_loss": total_loss / num_batches,
        }

        for key, value in metrics_accumulator.items():
            val_metrics[f"val_{key}"] = value / num_batches

        # Compute global causal metrics
        all_repr_benign = torch.cat(all_repr_benign, dim=0)
        all_repr_benign_cf = torch.cat(all_repr_benign_cf, dim=0)
        all_repr_injection = torch.cat(all_repr_injection, dim=0)

        causal_metrics = compute_causal_metrics(
            all_repr_benign,
            all_repr_benign_cf,
            all_repr_injection
        )

        # Add causal metrics (without val_ prefix for callbacks)
        val_metrics.update(causal_metrics)

        # Print validation summary
        print("\nValidation Results:")
        print(f"  Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Causal Stability: {causal_metrics['causal_stability']:.4f}")
        print(f"  Spurious Separation: {causal_metrics['spurious_separation']:.4f}")
        print(f"  Causal Discrimination: {causal_metrics['causal_discrimination']:.4f}")

        # Clear memory after validation
        clear_memory()

        return val_metrics

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save model checkpoint with ALL state.

        Args:
            output_dir: Directory to save to (defaults to self.output_dir)
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save causal projection
        if hasattr(self.model, 'causal_projection'):
            torch.save(
                self.model.causal_projection.state_dict(),
                save_dir / "causal_projection.pt"
            )

        # Save COMPLETE training state (use trainer_state.pt for consistency)
        torch.save({
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "batch_idx": self.current_batch_idx,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None
        }, save_dir / "trainer_state.pt")

        print(f"✓ Complete checkpoint saved to {save_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_dir: Directory with checkpoint
        """
        checkpoint_path = Path(checkpoint_dir)

        # Load model
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)

        # Load causal projection weights
        if hasattr(self.model, 'causal_projection'):
            projection_path = checkpoint_path / "causal_projection.pt"
            if projection_path.exists():
                self.model.causal_projection.load_state_dict(
                    torch.load(projection_path, map_location=self.device)
                )
                print(f"Loaded causal projection from {projection_path}")
            else:
                print("Warning: causal_projection.pt not found in checkpoint")

        # Load training state (saved by callbacks as trainer_state.pt)
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        if trainer_state_path.exists():
            trainer_state = torch.load(trainer_state_path, map_location=self.device)

            self.current_epoch = trainer_state.get("epoch", 0)
            self.global_step = trainer_state.get("global_step", 0)

            # Set resume tracking for within-epoch resumption
            self.resumed_epoch = self.current_epoch
            # Calculate batch_idx from global_step if not saved directly
            if "batch_idx" in trainer_state:
                self.resumed_batch_idx = trainer_state["batch_idx"]
            else:
                # Calculate from global_step: each step represents gradient_accumulation_steps batches
                # For within-epoch resumption, we need the batch index within the current epoch
                batches_per_epoch = len(self.train_dataloader)
                total_batches = self.global_step * self.gradient_accumulation_steps
                # If we're still in the first epoch (or any incomplete epoch), don't use modulo
                batches_before_current_epoch = self.current_epoch * batches_per_epoch
                self.resumed_batch_idx = total_batches - batches_before_current_epoch

            if self.optimizer and "optimizer_state" in trainer_state:
                self.optimizer.load_state_dict(trainer_state["optimizer_state"])

            if self.scheduler and "scheduler_state" in trainer_state:
                self.scheduler.load_state_dict(trainer_state["scheduler_state"])

            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}, batch {self.resumed_batch_idx}")
        else:
            print("Warning: trainer_state.pt not found, starting fresh")


if __name__ == "__main__":
    print("Trainer module loaded successfully")
    print("Use train.py to start training")
