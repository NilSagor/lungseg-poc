"""SegResNet baseline (implemented in Phase 14)."""

import torch
from torch import nn


class SegResNetSeg(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()
        raise NotImplementedError("SegResNetSeg is implemented in Phase 14")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError