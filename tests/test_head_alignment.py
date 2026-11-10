"""Regression: validation must index the feature cache through `feat_row`.

manifest_v1 is shorter than the cached feature file (G3 leakage guard drops test
rows), so using manifest row numbers directly as feature rows silently evaluates
the wrong images and pins the AUROC at chance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_head import evaluate  # noqa: E402


class IdentityHead(torch.nn.Module):
    def forward(self, x):
        return x[:, :9]


def test_evaluate_maps_manifest_rows_through_feat_row() -> None:
    n_rows = 8
    features = np.zeros((n_rows, 1024), dtype=np.float32)
    features[:, 0] = np.arange(n_rows, dtype=np.float32)

    feat_row = np.array([7, 3, 0, 5, 1, 6, 2, 4], dtype=np.int64)
    Y = np.zeros((n_rows, 9), dtype=np.float32)
    M = np.zeros((n_rows, 9), dtype=np.float32)
    for row in range(n_rows):
        label = int(feat_row[row] >= 4)
        Y[row, 0] = label
        M[row, 0] = 1

    rows = np.arange(n_rows)
    per_label, macro, used = evaluate(
        IdentityHead(), features, feat_row, Y, M, rows, torch.device("cpu"), min_pos=1, batch=4
    )
    assert used == ["normal"]
    assert per_label["normal"]["auroc"] == 1.0, (
        "feature rows are misaligned: manifest index was used instead of feat_row"
    )
    assert macro == 1.0
