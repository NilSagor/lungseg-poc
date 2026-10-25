"""
LightningModule and callbacks for the segmentation baselines (E0, E1, Phase 11-14).
"""
from __future__ import annotations

import math
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from monai.inferers import sliding_window_inference

from lungseg.diffusion import DiffusionSchedule
from lungseg.losses import BCEDiceLoss, BoundaryLoss, deep_supervision_loss
from lungseg.metrics import SegmentationMetrics
from lungseg.models import DiffusionRefiner, build_from_config


def build_loss(cfg: dict, use_boundary: bool):
    seg = BCEDiceLoss()
    bnd = BoundaryLoss() if use_boundary else None
    w = cfg["training"].get("boundary_weight", 0.0)

    def loss_fn(logits, target):
        loss = seg(logits, target)
        if bnd is not None:
            loss = loss + w * bnd(logits, target)
        return loss

    return loss_fn


class LitSegmentation(L.LightningModule):
    """Same objective, optimiser and validation protocol as the old loop:
    BCE+Dice (+ boundary), deep-supervision aware, AdamW, sliding-window
    validation on whole volumes."""

    def __init__(self, cfg: dict, use_boundary: bool = False):
        super().__init__()
        self.cfg = cfg
        self.model = build_from_config(cfg)
        self.loss_fn = build_loss(cfg, use_boundary)
        self.val_metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        loss = deep_supervision_loss(self.loss_fn, self.model(image), label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=image.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        e = self.cfg["evaluation"]
        logits = sliding_window_inference(
            inputs=batch["image"], roi_size=tuple(self.cfg["data"]["patch_size"]),
            sw_batch_size=e["sw_batch_size"], predictor=self.model, overlap=e["overlap"])
        self.val_metrics.update(logits, batch["label"])

    def on_validation_epoch_end(self):
        for name, value in self.val_metrics.aggregate().items():
            self.log(f"val/{name}", value)
        self.val_metrics.reset()

    def configure_optimizers(self):
        t = self.cfg["training"]
        return torch.optim.AdamW(
            self.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])


def pad_to_multiple(x, divisor):
    """Replicate-pad spatial dims to a multiple of ``divisor`` (replicate avoids edge artefacts)."""
    pads = []
    for size in reversed(x.shape[2:]):
        pads.extend([0, (divisor - size % divisor) % divisor])
    return F.pad(x, pads, mode="replicate") if sum(pads) else x


def crop_to_shape(x, original_shape):
    return x[(slice(None), slice(None)) + tuple(slice(0, s) for s in original_shape[2:])]


class LitDiffusion(L.LightningModule):
    """Conditional diffusion refiner on top of a frozen, already-trained baseline
    (E2 / E3). Only the refiner is optimised; the baseline stays in eval mode."""

    def __init__(self, cfg: dict, baseline: torch.nn.Module):
        super().__init__()
        self.cfg = cfg
        self.baseline = baseline.eval()
        for p in self.baseline.parameters():
            p.requires_grad_(False)
        self.refiner = DiffusionRefiner(
            channels=tuple(cfg["model"]["channels"]),
            num_res_units=cfg["model"]["num_res_units"])
        d = cfg["diffusion"]
        self.schedule = DiffusionSchedule(  # plain object, not an nn.Module: moved in setup()
            timesteps=d["timesteps"], beta_start=d["beta_start"], beta_end=d["beta_end"])
        self.val_metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])

    def setup(self, stage=None):
        self.schedule.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()  # Lightning calls .train() on everything; keep the baseline frozen
        return self

    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"].float()
        with torch.no_grad():
            initial_logits = self.baseline(image)
        noise = torch.randn_like(label)
        t = torch.randint(0, self.schedule.timesteps, (image.shape[0],), device=self.device)
        noisy = self.schedule.add_noise(label, noise, t)
        eps_hat = self.refiner(image, initial_logits, noisy, t)
        loss = self.cfg["training"]["diffusion_weight"] * F.mse_loss(eps_hat, noise)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=image.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        e, image = self.cfg["evaluation"], batch["image"]
        initial = sliding_window_inference(
            inputs=image, roi_size=tuple(self.cfg["data"]["patch_size"]),
            sw_batch_size=e["sw_batch_size"], predictor=self.baseline, overlap=e["overlap"])
        divisor = 2 ** (len(self.cfg["model"]["channels"]) - 1)
        refined = self.schedule.sample(
            refiner=self.refiner, image=pad_to_multiple(image, divisor),
            initial_logits=pad_to_multiple(initial, divisor),
            inference_steps=self.cfg["diffusion"]["inference_steps"])
        self.val_metrics.update(crop_to_shape(refined, image.shape), batch["label"])

    def on_validation_epoch_end(self):
        for name, value in self.val_metrics.aggregate().items():
            self.log(f"val/{name}", value)
        self.val_metrics.reset()

    def configure_optimizers(self):
        t = self.cfg["training"]
        return torch.optim.AdamW(
            self.refiner.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])


class PlainWeightsExport(Callback):
    """Write plain ``state_dict`` files that evaluate.py, train_diffusion.py,
    train_joint.py and the Makefile load: ``best.pt`` on every new best
    validation Dice, ``last.pt`` at the end of training."""

    def __init__(self, out_dir, monitor: str = "val/dice", best: float = -math.inf,
                 attr: str = "model", save_best: bool = True):
        self.out_dir = Path(out_dir)
        self.monitor = monitor
        self.attr = attr  # which submodule to export: "model" (segmentation) or "refiner"
        self.save_best = save_best
        self.best = best  # pass the previous best when resuming, so best.pt is never downgraded

    def on_validation_end(self, trainer, pl_module):
        score = trainer.callback_metrics.get(self.monitor)
        if trainer.sanity_checking or score is None or not self.save_best:
            return
        if float(score) > self.best:
            self.best = float(score)
            torch.save(getattr(pl_module, self.attr).state_dict(), self.out_dir / "best.pt")

    def on_fit_end(self, trainer, pl_module):
        torch.save(getattr(pl_module, self.attr).state_dict(), self.out_dir / "last.pt")


class HistoryRecorder(Callback):
    """One row per validation: epoch, train_loss, dice, iou, hd95 (same shape
    as the old history.json)."""

    def __init__(self, initial: list[dict] | None = None):
        self.rows: list[dict] = list(initial or [])  # earlier rows when resuming

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = {k.split("/", 1)[1]: float(v)
               for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
        self.rows.append({"epoch": trainer.current_epoch + 1, "train_loss": None, **val})

    def on_train_epoch_end(self, trainer, pl_module):
        # Validation runs before this hook, so fill in the epoch's train loss here.
        if self.rows and self.rows[-1]["epoch"] == trainer.current_epoch + 1:
            self.rows[-1]["train_loss"] = float(trainer.callback_metrics["train/loss"])