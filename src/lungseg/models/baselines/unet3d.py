"""Plain 3D U-Net baseline (the E0 reference)."""
import torch
from monai.networks.nets import UNet
from torch import nn


class BaselineUNet(nn.Module):
    """Compact 3D U-Net producing a single-channel tumor logit map."""

    def __init__(self, channels=(16, 32, 64, 128), num_res_units: int = 1):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=tuple(channels),
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)