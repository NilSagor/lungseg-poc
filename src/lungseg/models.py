"""3D U-Net baseline and conditional diffusion refiner."""

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class BaselineUNet(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), num_res_units=1):
        super().__init__()
        self.net = UNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            channels=tuple(channels),
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
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


class DiffusionRefiner(nn.Module):
    """Conditional noise-prediction network.
    Channels: [CT, initial_logits, noisy_mask, time_channel] -> predicted noise.
    """

    def __init__(self, channels=(16, 32, 64, 128), num_res_units=1, time_embed_dim=32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, 1)
        self.net = UNet(
            spatial_dims=3, in_channels=4, out_channels=1,
            channels=tuple(channels),
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
        )

    def forward(self, image, initial_logits, noisy_mask, timestep):
        emb = self.time_embed(timestep)
        t_scalar = self.time_proj(emb).view(-1, 1, 1, 1, 1)
        time_channel = t_scalar.expand(-1, -1, *image.shape[-3:])
        x = torch.cat([image, initial_logits, noisy_mask, time_channel], dim=1)
        return self.net(x)