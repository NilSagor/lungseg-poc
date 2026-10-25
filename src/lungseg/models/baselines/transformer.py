"""Transformer baselines (MONAI): SwinUNETR and UNETR.

Input-size rules
----------------
* SwinUNETR: every spatial dim must be a multiple of 32 and, in practice,
  >= 64. At 32 the deepest feature map is 1x1x1 and InstanceNorm rejects it
  in train mode. 96^3 patches are fine.
* UNETR: ``img_size`` is fixed at construction and must equal the patch /
  sliding-window ROI exactly. The factory reads it from ``data.patch_size``
  so the model and the inference ROI cannot disagree.
"""


from monai.networks.nets import UNETR, SwinUNETR
from torch import nn


class SwinUNETRSeg(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, use_checkpoint=True):
        super().__init__()
        self.net = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,   # must be divisible by 12
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

    def forward(self, x):
        return self.net(x)


class UNETRSeg(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=(96, 96, 96), feature_size=16):
        super().__init__()
        self.net = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=tuple(img_size),
            feature_size=feature_size,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            spatial_dims=3,
        )

    def forward(self, x):
        return self.net(x)