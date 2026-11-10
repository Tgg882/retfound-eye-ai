"""Portable path helpers for the CFP-9 release.

The frozen manifest uses absolute paths from the authoring machine.  A release
copy uses paths relative to the project ``dataset/`` directory.  These helpers
normalise both forms to the same stable key and resolve them against the local
checkout, so evaluation and inference do not depend on the author's drive
letters.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CFP9_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset"
DATA_ROOT = Path(os.environ.get("CFP9_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser().resolve()

# Longest / most specific markers first.  The key is the suffix beginning at
# the marker, e.g. ``01_MESSIDOR2/train/class_0/x.jpg``.
_DATA_MARKERS = (
    "_raw/Fundus10/images/original",
    "DDR-dataset/DR_grading",
    "01_MESSIDOR2",
    "02_APTOS2019",
    "03_IDRID",
    "04_Fundus10",
    "05_PAPILA",
)


def _slash(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def path_key(path: str | Path) -> str:
    """Return a machine-independent key for a raw, preprocessed, or cache path."""
    s = _slash(path)
    for marker in _DATA_MARKERS:
        idx = s.find(marker)
        if idx >= 0:
            return s[idx:].lstrip("/")
    for marker in ("outputs/", "dataset/"):
        idx = s.find(marker)
        if idx >= 0:
            return s[idx:].lstrip("/")
    return s.lstrip("./")


def resolve_data_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    """Resolve a raw-image path against ``CFP9_DATA_ROOT`` (default ``dataset/``)."""
    root = Path(data_root).expanduser().resolve() if data_root else DATA_ROOT
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    key = path_key(path)
    if key.startswith("dataset/"):
        return PROJECT_ROOT / key
    return root / key


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a preprocessed/output path against the repository root."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    return PROJECT_ROOT / path_key(path)


def display_path(path: str | Path) -> str:
    """Best-effort relative path for reports; never raises on external paths."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
