"""Evaluate the DistilVulSATD TinyBERT student on the public test set.

Default thresholds (0.20, 0.55) match the accepted e5 submission configuration.
Weights are large — download from the Drive folder linked in README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distilvulsatd.data.dataset import labels_to_array, load_competition_frames, save_json  # noqa: E402
from distilvulsatd.evaluate import evaluate_model, print_eval_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to student best_model/ (tokenizer + weights).",
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "NLBSE_2027_Dataset"))
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--thr-vuln", type=float, default=0.20)
    parser.add_argument("--thr-mat", type=float, default=0.55)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 512 rows, 1 repetition, no FLOPs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "student_eval.json"),
    )
    args = parser.parse_args()

    train_df, test_df = load_competition_frames(args.data_dir)
    texts = test_df["text"].tolist()
    labels = labels_to_array(test_df)

    if args.quick:
        texts = texts[:512]
        labels = labels[:512]
        repetitions = 1
        measure_flops = False
    else:
        repetitions = args.repetitions
        measure_flops = True

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = evaluate_model(
        model,
        tokenizer,
        texts,
        labels,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=(args.thr_vuln, args.thr_mat),
        repetitions=repetitions,
        measure_flops=measure_flops,
    )
    print_eval_result(result, title="DISTILVULSATD STUDENT")

    payload = result.to_dict()
    payload.update(
        {
            "model_dir": str(model_dir),
            "thr_vuln": args.thr_vuln,
            "thr_mat": args.thr_mat,
            "max_length": args.max_length,
            "quick": args.quick,
            "n_examples": len(texts),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
