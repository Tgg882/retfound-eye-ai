"""Dataset helpers for cached RETFound features (Stage 1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from paths import path_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((PROJECT_ROOT / "data_contract" / "labels.json").read_text(encoding="utf-8"))["labels"]
Y_COLS = [f"y_{label}" for label in LABELS]
M_COLS = [f"m_{label}" for label in LABELS]


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"path": str})
    if df["quality_flag"].isna().any():
        raise ValueError("manifest has NaN quality_flag; build it with --quality first")
    return df


def align_features(manifest: pd.DataFrame, feature_path: str | Path, index_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    index = pd.read_csv(index_path, dtype={"path": str})["path"].map(path_key)
    pos = manifest["path"].map(path_key).map({p: i for i, p in enumerate(index)})
    if pos.isna().any():
        raise ValueError(f"{int(pos.isna().sum())} manifest rows have no cached feature")
    return pos.to_numpy(dtype=np.int64)


class FeatureDataset(Dataset):
    """Cached-feature dataset; `rows` index the manifest, `feature_rows` index the npy."""

    def __init__(self, features, manifest: pd.DataFrame, rows: np.ndarray, feature_rows: np.ndarray) -> None:
        self.features = features
        self.manifest = manifest.reset_index(drop=True)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.feature_rows = np.asarray(feature_rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        rec = self.manifest.iloc[int(self.rows[i])]
        feat = torch.from_numpy(np.array(self.features[int(self.feature_rows[i])], dtype=np.float32, copy=True))
        y = torch.tensor(rec[Y_COLS].to_numpy(dtype=np.float32))
        m = torch.tensor(rec[M_COLS].to_numpy(dtype=np.float32))
        w = torch.tensor(float(rec["sample_weight"]), dtype=torch.float32)
        return feat, y, m, w
