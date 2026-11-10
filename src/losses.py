"""Masked BCE with label smoothing + per-label and per-sample weighting (plan 8.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_masked_bce(
    logits,
    target,
    mask,
    label_counts,
    sample_weight=None,
    eps: float = 0.05,
    alpha: float = 0.25,
):
    """Masked BCE.

    Per label the masked BCE is averaged over the *known* samples only.  When
    ``sample_weight`` is given (shape ``[B]``) each sample contributes in
    proportion to its weight, i.e. the average is weighted, not the final loss
    scaled by the batch mean.
    """
    t = target * (1 - eps) + 0.5 * eps
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")  # [B, L]

    if sample_weight is None:
        num = bce * mask
        den = mask.sum(dim=0)
    else:
        w = sample_weight.to(bce.dtype).view(-1, 1)
        num = bce * mask * w
        den = mask * w
        den = den.sum(dim=0)
    loss_c = num.sum(dim=0) / den.clamp_min(1e-6)

    w_label = label_counts.to(bce.dtype).pow(-alpha)
    w_label = w_label / w_label.sum()
    return (w_label * loss_c).sum()
