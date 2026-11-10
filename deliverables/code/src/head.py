"""9-label classification head (plan appendix A)."""
from __future__ import annotations

import torch.nn as nn


class CFP9Head(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 512, n: int = 9, p: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, n),
        )

    def forward(self, x):
        return self.net(x)
