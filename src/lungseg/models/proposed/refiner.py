"""Conditional diffusion refiner (noise-prediction network)."""
import torch
from monai.networks.nets import UNet
from torch import nn

from .time_embedding import SinusoidalTimeEmbedding


class DiffusionRefiner(nn.Module):
    """Conditional noise-prediction network.

    Inputs concatenated on channel axis:
        1. CT patch
        2. initial logits (baseline output)
        3. noisy mask y_t
        4. broadcast timestep channel

    Output: predicted noise epsilon, same spatial shape as input.
    """

    def __init__(
        self,
        channels=(16, 32, 64, 128),
        num_res_units: int = 1,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, 1)
        self.net = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=1,
            channels=tuple(channels),
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
        )

    def forward(
        self,
        image: torch.Tensor,
        initial_logits: torch.Tensor,
        noisy_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.time_embed(timestep)
        t_scalar = self.time_proj(emb).view(-1, 1, 1, 1, 1)
        time_channel = t_scalar.expand(-1, -1, *image.shape[-3:])
        x = torch.cat([image, initial_logits, noisy_mask, time_channel], dim=1)
        return self.net(x)