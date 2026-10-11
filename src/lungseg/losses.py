"""Segmentation and boundary losses."""

import torch.nn.functional as F
from monai.losses import DiceLoss
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True)

    def forward(self, logits, target):
        target = target.float()
        return self.dice(logits, target) + F.binary_cross_entropy_with_logits(logits, target)


def soft_erode(x):
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def soft_dilate(x):
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


class BoundaryLoss(nn.Module):
    """Distance-weighted BCE emphasising the tumor contour band."""

    def forward(self, logits, target):
        target = target.float()
        erode = soft_erode(target)
        dilate = soft_dilate(target)
        band = (dilate - erode).clamp(0, 1)
        weight = 1.0 + 4.0 * band
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def deep_supervision_loss(loss_fn, output, target):
    """Apply ``loss_fn`` to a model output that may carry several heads.
    Head i is weighted 0.5**i (1, 0.5, 0.25, ...) and the sum is normalised.
    """
    if isinstance(output, (list, tuple)):
        heads = list(output)
    elif output.ndim == 6:
        heads = list(output.unbind(dim=1))
    else:
        return loss_fn(output, target)
    weights = [0.5 ** i for i in range(len(heads))]
    total = sum(w * loss_fn(h, target) for w, h in zip(weights, heads))
    return total / sum(weights)