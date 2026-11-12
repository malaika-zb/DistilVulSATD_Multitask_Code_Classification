"""Official-style metrics, runtime/GFLOPs measurement, and submission score."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data.dataset import LABEL_NAMES

MAX_AVG_RUNTIME = 1000.0
MAX_AVG_FLOPS = 3_000_000.0


@dataclass
class ClassMetrics:
    cat: str
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


@dataclass
class EvalResult:
    scores: pd.DataFrame
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_runtime: float
    avg_flops: float
    submission_score: float
    predictions: np.ndarray
    labels: np.ndarray

    def to_dict(self) -> dict:
        return {
            "avg_precision": self.avg_precision,
            "avg_recall": self.avg_recall,
            "avg_f1": self.avg_f1,
            "avg_runtime": self.avg_runtime,
            "avg_flops": self.avg_flops,
            "submission_score": self.submission_score,
            "per_class": self.scores.to_dict(orient="records"),
        }


def submission_score(
    avg_f1: float,
    avg_runtime: float,
    avg_flops: float,
    max_avg_runtime: float = MAX_AVG_RUNTIME,
    max_avg_flops: float = MAX_AVG_FLOPS,
) -> float:
    """NLBSE'27 ranking formula (matches CodeBERT_baseline.ipynb)."""
    return (
        0.6 * avg_f1
        + 0.2 * max((max_avg_runtime - avg_runtime) / max_avg_runtime, 0.0)
        + 0.2 * max((max_avg_flops - avg_flops) / max_avg_flops, 0.0)
    )


def per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str] = LABEL_NAMES,
) -> List[ClassMetrics]:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    out: List[ClassMetrics] = []
    for class_idx, name in enumerate(label_names):
        tp = int(np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 1)))
        fp = int(np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 0)))
        tn = int(np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 0)))
        fn = int(np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out.append(
            ClassMetrics(
                cat=name,
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
        )
    return out


def metrics_from_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    avg_runtime: float = 0.0,
    avg_flops: float = 0.0,
) -> EvalResult:
    class_scores = per_class_metrics(predictions, labels)
    scores_df = pd.DataFrame([asdict(c) for c in class_scores])[
        ["cat", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    ]
    avg_f1 = float(scores_df["f1"].mean())
    avg_p = float(scores_df["precision"].mean())
    avg_r = float(scores_df["recall"].mean())
    return EvalResult(
        scores=scores_df,
        avg_precision=avg_p,
        avg_recall=avg_r,
        avg_f1=avg_f1,
        avg_runtime=avg_runtime,
        avg_flops=avg_flops,
        submission_score=submission_score(avg_f1, avg_runtime, avg_flops),
        predictions=np.asarray(predictions),
        labels=np.asarray(labels),
    )


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    labels: Union[Sequence[Sequence[int]], np.ndarray],
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 600,
    max_length: int = 512,
    threshold: Union[float, Sequence[float]] = 0.5,
    repetitions: int = 10,
    measure_flops: bool = True,
    show_progress: bool = True,
) -> EvalResult:
    """
    Run official-style evaluation with timed/profiled forward passes.

    Matches the competition baseline notebook protocol:
    AMP autocast, torch.profiler with_flops, repetitions averaged.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenized = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    if isinstance(threshold, (float, int)):
        thr = np.array([float(threshold), float(threshold)], dtype=np.float32)
    else:
        thr = np.asarray(threshold, dtype=np.float32)
        if thr.shape != (2,):
            raise ValueError("threshold must be a float or length-2 sequence")

    predictions: List[List[int]] = []
    total_time = 0.0
    total_flops = 0.0

    iterator = tqdm(loader, desc="Evaluating", disable=not show_progress)
    use_cuda = device.type == "cuda"

    for batch in iterator:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        begin = time.time()
        if use_cuda and measure_flops:
            with torch.amp.autocast("cuda"):
                with torch.profiler.profile(with_flops=True) as prof:
                    for _ in range(repetitions):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_flops += sum(k.flops for k in prof.key_averages()) / 1e9
        elif use_cuda:
            with torch.amp.autocast("cuda"):
                for _ in range(repetitions):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            for _ in range(repetitions):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_time += time.time() - begin

        logits = outputs.logits
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        batch_pred = (probs > thr.reshape(1, -1)).astype(np.int32).tolist()
        predictions.extend(batch_pred)

        del input_ids, attention_mask, outputs, logits
        if use_cuda:
            torch.cuda.empty_cache()

    labels_arr = np.asarray(labels, dtype=np.int64)
    pred_arr = np.asarray(predictions, dtype=np.int64)
    avg_runtime = total_time / repetitions
    avg_flops = total_flops / repetitions if measure_flops else 0.0
    return metrics_from_predictions(pred_arr, labels_arr, avg_runtime, avg_flops)


def print_eval_result(result: EvalResult, title: str = "RESULTS") -> None:
    print(f"----------------{title}--------------------")
    print(result.scores.to_string(index=False))
    print(f"avg precision: {result.avg_precision:.6f}")
    print(f"avg recall:    {result.avg_recall:.6f}")
    print(f"avg F1:        {result.avg_f1:.6f}")
    print(f"avg runtime:   {result.avg_runtime:.6f} s")
    print(f"avg GFLOPs:    {result.avg_flops:.6f}")
    print(f"submission:    {result.submission_score:.4f}")
