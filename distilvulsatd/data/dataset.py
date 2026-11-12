"""Dataset loading, label parsing, and train/val carving for NLBSE'27."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_NAMES = ["Vulnerability", "MAT"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "NLBSE_2027_Dataset"


def parse_label(value: Union[str, list, tuple, np.ndarray]) -> List[int]:
    """Parse Label field from HF list or CSV string like '[0 1]' / '[0, 1]'."""
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [int(x) for x in list(value)]
        if len(vals) != 2:
            raise ValueError(f"Expected 2-label vector, got {vals!r}")
        return vals

    text = str(value).strip()
    # Prefer robust numeric extraction over eval.
    nums = re.findall(r"-?\d+", text)
    if len(nums) >= 2:
        return [int(nums[0]), int(nums[1])]

    try:
        parsed = ast.literal_eval(text.replace(" ", ","))
        return [int(parsed[0]), int(parsed[1])]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot parse label: {value!r}") from exc


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"Function", "Label"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in dataframe")
    labels = out["Label"].apply(parse_label)
    out["vuln"] = labels.apply(lambda x: int(x[0]))
    out["satd"] = labels.apply(lambda x: int(x[1]))
    out["label_list"] = labels
    out["Function"] = out["Function"].fillna("").astype(str)
    return out


def load_competition_frames(
    data_dir: Optional[Union[str, Path]] = None,
    prefer_local: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load train/test frames.

    Prefers local CSVs under NLBSE_2027_Dataset; falls back to Hugging Face.
    """
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    frames: Dict[str, pd.DataFrame] = {}

    if prefer_local and root.exists():
        for split in ("train", "test"):
            path = root / f"{split}.csv"
            if path.exists():
                frames[split] = _normalize_frame(pd.read_csv(path))
        if "train" in frames and "test" in frames:
            return frames

    from datasets import load_dataset

    ds = load_dataset("NLBSE/nlbse27-code-comment-classification")
    for split in ds.keys():
        frames[split] = _normalize_frame(ds[split].to_pandas())
    return frames


def carve_train_val(
    train_df: pd.DataFrame,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified carve from train only (never touch public test for tuning).

    Stratifies on joint (vuln, satd) label string.
    """
    joint = train_df["vuln"].astype(str) + "_" + train_df["satd"].astype(str)
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=joint,
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def summarize_label_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts/rates for joint and marginal labels."""
    n = len(df)
    rows = []
    for v, s in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        count = int(((df["vuln"] == v) & (df["satd"] == s)).sum())
        rows.append(
            {
                "pattern": f"[{v},{s}]",
                "count": count,
                "rate": count / n if n else 0.0,
            }
        )
    rows.append(
        {
            "pattern": "vuln+",
            "count": int(df["vuln"].sum()),
            "rate": float(df["vuln"].mean()) if n else 0.0,
        }
    )
    rows.append(
        {
            "pattern": "satd+",
            "count": int(df["satd"].sum()),
            "rate": float(df["satd"].mean()) if n else 0.0,
        }
    )
    return pd.DataFrame(rows)


def labels_to_array(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["label_list"].to_numpy()).astype(np.int64)


def save_json(obj: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
