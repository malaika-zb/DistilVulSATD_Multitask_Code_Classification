"""Multitask classification wrappers for teacher/student experiments."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


def load_sequence_classifier(
    model_name: str,
    num_labels: int = 2,
    problem_type: str = "multi_label_classification",
) -> AutoModelForSequenceClassification:
    """Standard HF multi-label head (baseline-compatible)."""
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        id2label={0: "weakness", 1: "MAT"},
        label2id={"weakness": 0, "MAT": 1},
    )
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)


class DualHeadMultitaskModel(nn.Module):
    """
    Shared encoder + two binary heads (still one submitted model).

    Useful ablation vs a single 2-logit multi-label head.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.vuln_head = nn.Linear(hidden, 1)
        self.satd_head = nn.Linear(hidden, 1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = torch.cat([self.vuln_head(cls), self.satd_head(cls)], dim=-1)
        return type("Out", (), {"logits": logits})()
