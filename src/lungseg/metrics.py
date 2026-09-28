"""Metric collection for tumor segmentation."""
import torch

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU 


class SegmentationMetrics:
    def __init__(self, threshold=0.5, include_background=False):
        self.threshold = threshold
        self.dice = DiceMetric(include_background=include_background, reduction="mean")
        self.iou = MeanIoU(include_background=include_background, reduction="mean")
        self.hd95 = HausdorffDistanceMetric(
            include_background=include_background, percentile=95, reduction="mean")

    def update(self, logits, target):
        pred = (torch.sigmoid(logits) > self.threshold).float()
        target = (target > 0).float()
        self.dice(y_pred=pred, y=target)
        self.iou(y_pred=pred, y=target)
        for b in range(pred.shape[0]):
            if pred[b].sum() > 0 and target[b].sum() > 0:
                self.hd95(y_pred=pred[b:b+1], y=target[b:b+1])

    def aggregate(self):
        out = {"dice": float(self.dice.aggregate().item()),
               "iou": float(self.iou.aggregate().item())}
        try:
            out["hd95"] = float(self.hd95.aggregate().item())
        except Exception:
            out["hd95"] = float("nan")
        return out

    def reset(self):
        self.dice.reset(); self.iou.reset(); self.hd95.reset()