"""Transformer baselines: SwinUNETR and UNETR (implemented in Phase 13)."""

import torch
from torch import nn


class SwinUNETRSeg(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()
        raise NotImplementedError("SwinUNETRSeg is implemented in Phase 13")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class UNETRSeg(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()
        raise NotImplementedError("UNETRSeg is implemented in Phase 13")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError