"""nnU-Net-style baseline (implemented in Phase 12)."""
import torch
from torch import nn


class NNUNetStyle(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()
        raise NotImplementedError("NNUNetStyle is implemented in Phase 12")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError