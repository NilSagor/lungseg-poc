"""Sinusoidal timestep embedding used by the diffusion refiner."""
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional embedding of the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb