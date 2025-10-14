"""
Causal Contrastive Loss for LLM Security

Implements the loss function described in theory/causal_formalization.md

Goal:
- Maximize similarity: output(S, U_benign) ≈ output(S, U_benign')
- Minimize similarity: output(S, U_benign) ≠ output(S, U_injection)

This enforces causal invariance to benign changes while maintaining sensitivity to injections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CausalContrastiveLoss(nn.Module):
    """
    Causal contrastive loss for training robust LLMs.

    Loss = -causal_stability + spurious_separation + lambda * task_loss

    Where:
        - causal_stability: similarity(repr(S, U_benign), repr(S, U_benign'))
        - spurious_separation: -similarity(repr(S, U_benign), repr(S, U_injection))
        - task_loss: standard cross-entropy for correct output generation
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_task: float = 1.0,
        lambda_causal: float = 0.5,
        lambda_spurious: float = 0.5,
        similarity_metric: str = "cosine"
    ):
        """
        Initialize the causal contrastive loss.

        Args:
            temperature: Temperature for contrastive learning
            lambda_task: Weight for task loss (correct generation)
            lambda_causal: Weight for causal stability term
            lambda_spurious: Weight for spurious separation term
            similarity_metric: "cosine" or "dot_product"
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_task = lambda_task
        self.lambda_causal = lambda_causal
        self.lambda_spurious = lambda_spurious
        self.similarity_metric = similarity_metric

    def forward(
        self,
        repr_benign: torch.Tensor,
        repr_benign_counterfactual: torch.Tensor,
        repr_injection: torch.Tensor,
        logits_benign: Optional[torch.Tensor] = None,
        labels_benign: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the causal contrastive loss.

        Args:
            repr_benign: Representation for (S, U_benign) [batch, hidden_dim]
            repr_benign_counterfactual: Representation for (S, U_benign') [batch, hidden_dim]
            repr_injection: Representation for (S, U_injection) [batch, hidden_dim]
            logits_benign: Output logits for benign input [batch, seq_len, vocab_size]
            labels_benign: Target labels for benign input [batch, seq_len]

        Returns:
            Dictionary containing:
                - loss: Total loss
                - causal_stability: Causal stability term
                - spurious_separation: Spurious separation term
                - task_loss: Task loss (if provided)
        """
        # Normalize representations
        repr_benign = F.normalize(repr_benign, dim=-1)
        repr_benign_counterfactual = F.normalize(repr_benign_counterfactual, dim=-1)
        repr_injection = F.normalize(repr_injection, dim=-1)

        # Compute similarities
        sim_benign_benign = self._compute_similarity(repr_benign, repr_benign_counterfactual)
        sim_benign_injection = self._compute_similarity(repr_benign, repr_injection)

        # Causal stability: encourage benign variants to be similar
        # Higher similarity = lower loss
        causal_stability = -torch.mean(sim_benign_benign / self.temperature)

        # Spurious separation: encourage benign and injection to be different
        # Higher similarity = higher loss
        spurious_separation = torch.mean(sim_benign_injection / self.temperature)

        # Combine contrastive terms
        contrastive_loss = (
            self.lambda_causal * causal_stability +
            self.lambda_spurious * spurious_separation
        )

        # Task loss (standard cross-entropy for correct generation)
        task_loss_value = torch.tensor(0.0, device=repr_benign.device)
        if logits_benign is not None and labels_benign is not None:
            task_loss_value = self._compute_task_loss(logits_benign, labels_benign)

        # Total loss
        total_loss = contrastive_loss + self.lambda_task * task_loss_value

        return {
            "loss": total_loss,
            "causal_stability": -causal_stability,  # Return positive value for logging
            "spurious_separation": spurious_separation,
            "task_loss": task_loss_value,
            "contrastive_loss": contrastive_loss
        }

    def _compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two representation tensors.

        Args:
            x: First tensor [batch, hidden_dim]
            y: Second tensor [batch, hidden_dim]

        Returns:
            Similarity scores [batch]
        """
        if self.similarity_metric == "cosine":
            # Cosine similarity (assumes normalized inputs)
            return torch.sum(x * y, dim=-1)
        elif self.similarity_metric == "dot_product":
            return torch.sum(x * y, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def _compute_task_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard cross-entropy task loss.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]

        Returns:
            Task loss (scalar)
        """
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Ignore padding tokens (usually -100)
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction="mean"
        )

        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss variant for causal learning.

    This is an alternative formulation using the InfoNCE objective:
        L = -log[exp(sim(anchor, positive)) / sum_i exp(sim(anchor, negative_i))]
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_task: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_task = lambda_task

    def forward(
        self,
        repr_benign: torch.Tensor,
        repr_benign_counterfactual: torch.Tensor,
        repr_injection: torch.Tensor,
        logits_benign: Optional[torch.Tensor] = None,
        labels_benign: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss.

        Args:
            repr_benign: Anchor representation [batch, hidden_dim]
            repr_benign_counterfactual: Positive representation [batch, hidden_dim]
            repr_injection: Negative representation [batch, hidden_dim]
            logits_benign: Output logits [batch, seq_len, vocab_size]
            labels_benign: Target labels [batch, seq_len]

        Returns:
            Loss dictionary
        """
        # Normalize
        repr_benign = F.normalize(repr_benign, dim=-1)
        repr_benign_counterfactual = F.normalize(repr_benign_counterfactual, dim=-1)
        repr_injection = F.normalize(repr_injection, dim=-1)

        # Compute similarities
        sim_positive = torch.sum(repr_benign * repr_benign_counterfactual, dim=-1)
        sim_negative = torch.sum(repr_benign * repr_injection, dim=-1)

        # InfoNCE loss
        # log[exp(sim_pos) / (exp(sim_pos) + exp(sim_neg))]
        logits_contrastive = torch.stack([sim_positive, sim_negative], dim=1) / self.temperature
        labels_contrastive = torch.zeros(repr_benign.size(0), dtype=torch.long, device=repr_benign.device)

        contrastive_loss = F.cross_entropy(logits_contrastive, labels_contrastive)

        # Task loss
        task_loss_value = torch.tensor(0.0, device=repr_benign.device)
        if logits_benign is not None and labels_benign is not None:
            batch_size, seq_len, vocab_size = logits_benign.shape
            logits_flat = logits_benign.view(-1, vocab_size)
            labels_flat = labels_benign.view(-1)
            task_loss_value = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,
                reduction="mean"
            )

        total_loss = contrastive_loss + self.lambda_task * task_loss_value

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "task_loss": task_loss_value
        }


class TripletLoss(nn.Module):
    """
    Triplet loss variant for causal learning.

    L = max(0, margin + sim(anchor, negative) - sim(anchor, positive))
    """

    def __init__(
        self,
        margin: float = 1.0,
        lambda_task: float = 1.0
    ):
        super().__init__()
        self.margin = margin
        self.lambda_task = lambda_task

    def forward(
        self,
        repr_benign: torch.Tensor,
        repr_benign_counterfactual: torch.Tensor,
        repr_injection: torch.Tensor,
        logits_benign: Optional[torch.Tensor] = None,
        labels_benign: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute triplet loss."""
        # Normalize
        repr_benign = F.normalize(repr_benign, dim=-1)
        repr_benign_counterfactual = F.normalize(repr_benign_counterfactual, dim=-1)
        repr_injection = F.normalize(repr_injection, dim=-1)

        # Compute distances (use negative similarity as distance)
        dist_positive = 1 - torch.sum(repr_benign * repr_benign_counterfactual, dim=-1)
        dist_negative = 1 - torch.sum(repr_benign * repr_injection, dim=-1)

        # Triplet loss
        triplet_loss = torch.mean(
            F.relu(self.margin + dist_positive - dist_negative)
        )

        # Task loss
        task_loss_value = torch.tensor(0.0, device=repr_benign.device)
        if logits_benign is not None and labels_benign is not None:
            batch_size, seq_len, vocab_size = logits_benign.shape
            logits_flat = logits_benign.view(-1, vocab_size)
            labels_flat = labels_benign.view(-1)
            task_loss_value = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,
                reduction="mean"
            )

        total_loss = triplet_loss + self.lambda_task * task_loss_value

        return {
            "loss": total_loss,
            "triplet_loss": triplet_loss,
            "task_loss": task_loss_value
        }


if __name__ == "__main__":
    # Test the loss functions
    batch_size = 4
    hidden_dim = 768

    # Create dummy representations
    repr_benign = torch.randn(batch_size, hidden_dim)
    repr_benign_cf = torch.randn(batch_size, hidden_dim)
    repr_injection = torch.randn(batch_size, hidden_dim)

    # Test CausalContrastiveLoss
    loss_fn = CausalContrastiveLoss()
    losses = loss_fn(repr_benign, repr_benign_cf, repr_injection)

    print("CausalContrastiveLoss Test:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("\nAll loss functions initialized successfully!")
