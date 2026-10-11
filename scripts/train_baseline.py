"""Train a 3D segmentation baseline."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lungseg.config import load_config
from lungseg.data import build_dataloaders
from lungseg.losses import BCEDiceLoss, BoundaryLoss, deep_supervision_loss
from lungseg.metrics import SegmentationMetrics
from lungseg.models import build_from_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed)


def build_loss(cfg, use_boundary):
    seg = BCEDiceLoss()
    bnd = BoundaryLoss() if use_boundary else None
    w = cfg["training"]["boundary_weight"]

    def loss_fn(logits, target):
        loss = seg(logits, target)
        if bnd is not None:
            loss = loss + w * bnd(logits, target)
        return loss
    return loss_fn


def step(model, batch, optimizer, loss_fn, device, scaler=None, use_amp=False):
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = deep_supervision_loss(loss_fn, model(image), label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = deep_supervision_loss(loss_fn, model(image), label)
        loss.backward()
        optimizer.step()
    return loss.item()


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp, show_progress=True):
    model.train()
    running = 0.0
    it = tqdm(loader, desc="train", leave=False) if show_progress else loader
    for batch in it:
        running += step(model, batch, optimizer, loss_fn, device, scaler, use_amp)
    return running / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
    roi = tuple(cfg["data"]["patch_size"])
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = sliding_window_inference(
            inputs=image, roi_size=roi,
            sw_batch_size=cfg["evaluation"]["sw_batch_size"],
            predictor=model, overlap=cfg["evaluation"]["overlap"])
        metrics.update(logits, label)
    return metrics.aggregate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--use-boundary", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed-override", type=int, default=42)
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.seed_override is not None:
        cfg["seed"] = a.seed_override
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])

    if a.debug:
        subset = []
        for i, batch in enumerate(train_loader):
            subset.append(batch)
            if i >= 3:
                break
        train_loader = subset

    model = build_from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model'].get('name', 'unet3d')} ({n_params / 1e6:.2f}M params)")

    loss_fn = build_loss(cfg, use_boundary=a.use_boundary)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler(
        enabled=cfg["training"]["amp"] and device.type == "cuda")

    out_dir = Path(a.out or f"outputs/baseline{'_boundary' if a.use_boundary else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    epochs = a.max_epochs or cfg["training"]["baseline_epochs"]
    best_dice = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        if a.debug:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, None, False,
                show_progress=False)
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler,
                cfg["training"]["amp"])
        writer.add_scalar("loss/train", train_loss, epoch)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f}")

        if not a.debug and epoch % cfg["training"]["val_interval"] == 0:
            val_metrics = validate(model, val_loader, cfg, device)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(f"          val: {val_metrics}")
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if not a.debug and history:
        best = max(history, key=lambda r: r["dice"])
        (out_dir / "metrics.json").write_text(json.dumps({
            **best,
            "seed": cfg["seed"],
            "use_boundary": bool(a.use_boundary),
            "model": cfg["model"].get("name", "unet3d"),
            "n_params": n_params,
            "checkpoint": "best.pt",
        }, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()