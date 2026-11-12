"""Phase 0: reproduce official CodeBERT baseline on public test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distilvulsatd.data.dataset import labels_to_array, load_competition_frames, save_json  # noqa: E402
from distilvulsatd.evaluate import evaluate_model, print_eval_result  # noqa: E402

BASELINE_MODEL = "NLBSE/nlbse27-code-comment-classification-baseline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "NLBSE_2027_Dataset"))
    parser.add_argument("--model", type=str, default=BASELINE_MODEL)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 512 rows, 1 repetition, no FLOPs (CPU/GPU friendly).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "baseline_eval.json"),
    )
    args = parser.parse_args()

    frames = load_competition_frames(args.data_dir)
    test_df = frames["test"]
    if args.quick:
        test_df = test_df.head(512).reset_index(drop=True)
        repetitions = 1
        measure_flops = False
        batch_size = min(32, args.batch_size)
        print("QUICK MODE: subset + no FLOPs profiling")
    else:
        repetitions = args.repetitions
        measure_flops = True
        batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  n={len(test_df)}  model={args.model}")

    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    model = RobertaForSequenceClassification.from_pretrained(args.model)

    result = evaluate_model(
        model,
        tokenizer,
        test_df["Function"].tolist(),
        labels_to_array(test_df),
        device=device,
        batch_size=batch_size,
        max_length=args.max_length,
        repetitions=repetitions,
        measure_flops=measure_flops and device.type == "cuda",
    )
    print_eval_result(result, title="BASELINE")

    payload = {
        "model": args.model,
        "quick": args.quick,
        "n_examples": len(test_df),
        "device": str(device),
        "batch_size": batch_size,
        "max_length": args.max_length,
        "repetitions": repetitions,
        **result.to_dict(),
        "reference_full_test": {
            "avg_f1": 0.910186,
            "avg_runtime": 192.098,
            "avg_flops": 2666509.713,
            "submission_score": 0.7299,
        },
    }
    save_json(payload, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
