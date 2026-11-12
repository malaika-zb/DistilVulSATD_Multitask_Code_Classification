"""Losses for imbalanced multitask training and distillation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss for multi-label classification (TraCC-style).

    gamma_pos for positives, gamma_neg for negatives; optional per-label weights.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pt_pos = probs
        pt_neg = 1.0 - probs

        loss_pos = -((1 - pt_pos) ** self.gamma_pos) * torch.log(pt_pos.clamp_min(1e-8))
        loss_neg = -((1 - pt_neg) ** self.gamma_neg) * torch.log(pt_neg.clamp_min(1e-8))
        loss = targets * loss_pos + (1 - targets) * loss_neg

        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, -1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    hard_loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Soft teacher targets + hard labels for multi-label sigmoid outputs.

    soft: BCE between student probs and teacher softened probs
    hard: asymmetric focal / BCE on ground truth
    """
    hard_loss_fn = hard_loss_fn or nn.BCEWithLogitsLoss()
    soft_targets = torch.sigmoid(teacher_logits / temperature)
    student_soft = torch.sigmoid(student_logits / temperature)
    soft = F.binary_cross_entropy(student_soft.clamp(1e-8, 1 - 1e-8), soft_targets.detach())
    soft = soft * (temperature**2)
    hard = hard_loss_fn(student_logits, targets.float())
    return alpha * soft + (1.0 - alpha) * hard


def inverse_freq_weights(
    vuln_pos_rate: float,
    satd_pos_rate: float,
    satd_boost: float = 1.5,
) -> torch.Tensor:
    """sqrt(inverse frequency) style weights with optional SATD boost."""
    w_vuln = (1.0 / max(vuln_pos_rate, 1e-6)) ** 0.5
    w_satd = ((1.0 / max(satd_pos_rate, 1e-6)) ** 0.5) * satd_boost
    # normalize so mean weight ~ 1
    weights = torch.tensor([w_vuln, w_satd], dtype=torch.float32)
    return weights / weights.mean()
