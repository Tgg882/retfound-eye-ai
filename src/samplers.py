"""sqrt-balanced dataset sampler (plan 8.4 / appendix D)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def sqrt_balanced_sampler(manifest: pd.DataFrame, rows: np.ndarray, seed: int = 42) -> WeightedRandomSampler:
    """Dataset-balanced sampler: each dataset contributes ~1/sqrt(n) of the batches."""
    datasets = manifest.iloc[rows]["dataset"]
    sizes = datasets.value_counts().to_dict()
    w_ds = {name: 1.0 / math.sqrt(n) for name, n in sizes.items()}
    weights = torch.tensor([w_ds[d] / sizes[d] for d in datasets], dtype=torch.double)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
