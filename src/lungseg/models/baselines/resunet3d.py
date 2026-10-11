"""3D Residual U-Net with pre-activation residual blocks."""


import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    """Pre-activation block: IN -> act -> conv -> IN -> act -> conv, plus skip."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_ch, affine=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + identity


class _Stage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2, stride=1):
        super().__init__()
        blocks = [ResidualBlock3D(in_ch, out_ch, stride=stride)]
        blocks += [ResidualBlock3D(out_ch, out_ch) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class _DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.blocks = _Stage(out_ch + skip_ch, out_ch, n_blocks=n_blocks)

    def forward(self, x, skip):
        return self.blocks(torch.cat([self.up(x), skip], dim=1))


class ResUNet3D(nn.Module):
    """channels = feature width per level; input size must be divisible by
    2**(len(channels)-1)."""

    def __init__(self, channels=(16, 32, 64, 128), n_blocks=2, in_channels=1, out_channels=1):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("ResUNet3D needs at least two levels")
        channels = tuple(channels)
        self.stem = nn.Conv3d(in_channels, channels[0], 3, padding=1, bias=False)

        self.encoders = nn.ModuleList()
        prev = channels[0]
        for i, c in enumerate(channels):
            self.encoders.append(_Stage(prev, c, n_blocks, stride=1 if i == 0 else 2))
            prev = c

        self.decoders = nn.ModuleList(
            _DecoderStage(channels[i + 1], channels[i], channels[i], n_blocks)
            for i in reversed(range(len(channels) - 1))
        )
        # Pre-activation nets need a closing norm+act before the 1x1 head.
        self.final_norm = nn.InstanceNorm3d(channels[0], affine=True)
        self.final_act = nn.LeakyReLU(0.01, inplace=True)
        self.head = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = skips.pop()  # bottleneck
        for dec in self.decoders:
            x = dec(x, skips.pop())
        return self.head(self.final_act(self.final_norm(x)))