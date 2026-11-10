"""Regression tests for the Stage 1 loss / sampler fixes (round 2, item 4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from losses import balanced_masked_bce  # noqa: E402
from samplers import sqrt_balanced_sampler  # noqa: E402


def _logits(n=4, labels=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, labels, generator=g)


def test_masked_slots_do_not_affect_loss() -> None:
    logits = _logits()
    y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    counts = torch.tensor([2.0, 3.0, 2.0])
    loss_a = balanced_masked_bce(logits, y, m, counts)
    y2 = y.clone()
    y2[m == 0] = 1.0 - y2[m == 0]
    loss_b = balanced_masked_bce(logits, y2, m, counts)
    assert torch.equal(loss_a, loss_b)


def test_sample_weight_is_per_sample_not_a_global_scalar() -> None:
    logits = _logits(n=2, labels=1, seed=1)
    y = torch.tensor([[1.0], [0.0]])
    m = torch.ones_like(y)
    counts = torch.tensor([1.0])

    w = torch.tensor([4.0, 1.0])
    loss_w = balanced_masked_bce(logits, y, m, counts, sample_weight=w, eps=0.0, alpha=0.0)

    # hand-computed weighted masked BCE with the same smoothing/weighting rules
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    expected = (bce * m * w.view(-1, 1)).sum() / (m * w.view(-1, 1)).sum()
    assert torch.allclose(loss_w, expected, atol=1e-6)

    # flipping which sample carries the weight must change the loss
    loss_flipped = balanced_masked_bce(logits, y, m, counts, sample_weight=torch.tensor([1.0, 4.0]), eps=0.0, alpha=0.0)
    assert not torch.allclose(loss_w, loss_flipped)
    # and it must differ from the plain mean (the old `* sample_weight.mean()` behaviour)
    loss_plain = balanced_masked_bce(logits, y, m, counts, eps=0.0, alpha=0.0)
    assert not torch.allclose(loss_w, loss_plain)


def test_sqrt_balanced_sampler_is_seeded_with_torch_generator() -> None:
    df = pd.DataFrame({"dataset": ["A"] * 6 + ["B"] * 20})
    rows = np.arange(len(df))
    first = list(sqrt_balanced_sampler(df, rows, seed=7))
    second = list(sqrt_balanced_sampler(df, rows, seed=7))
    assert first == second
    assert first != list(sqrt_balanced_sampler(df, rows, seed=8))
