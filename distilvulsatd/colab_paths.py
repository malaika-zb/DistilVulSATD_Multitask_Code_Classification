"""
Persistent Google Drive paths for Colab.

Nothing important should stay only in /content — that dies with the session.
All checkpoints, metrics, and ablations go under DRIVE_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Change this if you put the folder somewhere else on Drive.
DEFAULT_DRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive/NLBSE27_DistilVulSATD"),
    Path("/content/drive/MyDrive/Multitask Code Classification/distilvulsatd_outputs"),
]


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "google.colab not available. Run this on Colab, or pass an explicit out_dir."
        ) from exc
    drive.mount("/content/drive")


def resolve_drive_root(
    preferred: Optional[str | Path] = None,
    create: bool = True,
) -> Path:
    """
    Return the persistent project output root on Google Drive.

    Layout:
      DRIVE_ROOT/
        results/
        checkpoints/
        logs/
    """
    if preferred is not None:
        root = Path(preferred)
    else:
        env = os.environ.get("NLBSE27_DRIVE_ROOT")
        if env:
            root = Path(env)
        else:
            root = DEFAULT_DRIVE_CANDIDATES[0]

    if create:
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
    return root


def setup_colab_drive(preferred: Optional[str | Path] = None) -> Path:
    """Mount Drive (if needed) and return persistent output root."""
    if not Path("/content/drive/MyDrive").exists():
        mount_drive()
    root = resolve_drive_root(preferred=preferred, create=True)
    print("DRIVE_ROOT =", root)
    print("results     =", root / "results")
    print("checkpoints =", root / "checkpoints")
    return root
