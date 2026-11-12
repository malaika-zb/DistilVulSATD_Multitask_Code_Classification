"""Phase 1: label balance, length stats, comment extraction coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distilvulsatd.data.dataset import (  # noqa: E402
    carve_train_val,
    load_competition_frames,
    save_json,
    summarize_label_balance,
)
from distilvulsatd.data.preprocess import extract_comments_and_code, format_input  # noqa: E402


def length_stats(series: pd.Series) -> dict:
    lengths = series.str.len()
    return {
        "n": int(len(series)),
        "mean_chars": float(lengths.mean()),
        "median_chars": float(lengths.median()),
        "p90_chars": float(lengths.quantile(0.90)),
        "p99_chars": float(lengths.quantile(0.99)),
        "max_chars": int(lengths.max()),
    }


def comment_coverage(df: pd.DataFrame, sample_size: int = 5000, seed: int = 42) -> dict:
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    has_comment = 0
    comment_lens = []
    code_lens = []
    for text in sample["Function"]:
        comments, code = extract_comments_and_code(text)
        if comments:
            has_comment += 1
        comment_lens.append(len(comments))
        code_lens.append(len(code))
    n = len(sample)
    return {
        "sample_size": n,
        "frac_with_extracted_comments": has_comment / n if n else 0.0,
        "mean_comment_chars": float(np.mean(comment_lens)) if comment_lens else 0.0,
        "mean_code_chars": float(np.mean(code_lens)) if code_lens else 0.0,
        "bimodal_example": format_input(sample.iloc[0]["Function"], mode="bimodal")[:300],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "NLBSE_2027_Dataset"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "data_analysis.json"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    frames = load_competition_frames(args.data_dir)
    train_df = frames["train"]
    test_df = frames["test"]
    fit_df, val_df = carve_train_val(train_df, val_fraction=args.val_fraction)

    satd_df = train_df[train_df["satd"] == 1]
    report = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "fit_size": len(fit_df),
        "val_size": len(val_df),
        "train_balance": summarize_label_balance(train_df).to_dict(orient="records"),
        "test_balance": summarize_label_balance(test_df).to_dict(orient="records"),
        "val_balance": summarize_label_balance(val_df).to_dict(orient="records"),
        "train_length": length_stats(train_df["Function"]),
        "test_length": length_stats(test_df["Function"]),
        "comment_coverage_train_sample": comment_coverage(train_df),
        "comment_coverage_satd_all": comment_coverage(satd_df, sample_size=len(satd_df)),
        "recommended_max_length_start": 256,
        "notes": [
            "Tune only on carved val from train; report once on public test.",
            "SATD (MAT) is the rare label and main F1 bottleneck.",
            "Overall comment extraction ~14%, but SATD+ rows are ~90%+ — bimodal packing is SATD-relevant.",
        ],
    }

    out_path = Path(args.out)
    save_json(report, out_path)
    print(json.dumps(report, indent=2)[:4000])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
