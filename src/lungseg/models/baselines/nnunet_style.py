"""
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation (2000)

Architecturally inspired by nnU-Net (3x3x3 kernels, stride-2 downsampling,
instance norm, leaky ReLU, residual blocks, deep supervision). It does NOT
include nnU-Net's data fingerprinting, auto-configuration or 5-fold
ensembling, so report it as "nnU-Net-style", not "nnU-Net".

Output contract
---------------
* train mode + deep supervision: ONE stacked tensor of shape
  (B, n_heads, C, D, H, W). MONAI already resamples every head to full
  resolution, so no target downsampling is needed. Consume it with
  ``lungseg.losses.deep_supervision_loss``.
* eval mode: a plain (B, C, D, H, W) tensor, so sliding-window inference
  and every other script work unchanged.
"""


from monai.networks.nets import DynUNet
from torch import nn


class NNUNetStyle(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features=None,
        channels=None,
        patch_size=(96, 96, 96),
        deep_supervision: bool = True,
    ):
        super().__init__()
        # Accept either 'features' or 'channels' for compatibility between tests and factory
        if features is not None:
            feature_list = features
        elif channels is not None:
            feature_list = channels
        else:
            feature_list = (16, 32, 64, 128)

        depth = len(feature_list)
        if depth < 3:
            raise ValueError("NNUNetStyle needs at least three levels")
            
        self.net = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3, 3, 3]] * depth,
            strides=[[1, 1, 1]] + [[2, 2, 2]] * (depth - 1),
            upsample_kernel_size=[[2, 2, 2]] * (depth - 1),
            filters=tuple(feature_list),
            norm_name="instance",
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=deep_supervision,
            deep_supr_num=min(2, depth - 2),
            res_block=True,
        )

    def forward(self, x):
        return self.net(x)