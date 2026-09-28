"""Train the 3D U-Net baseline (E0 / E1)."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lungseg.data import build_dataloaders
from lungseg.losses import BCEDiceLoss, BoundaryLoss
from lungseg.metrics import SegmentationMetrics
from lungseg.models import BaselineUNet


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


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp):
    model.train(); running = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(image); loss = loss_fn(logits, label)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(image); loss = loss_fn(logits, label)
            loss.backward(); optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
    roi = tuple(cfg["data"]["patch_size"])
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device); label = batch["label"].to(device)
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
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])

    if a.debug:
        subset = []
        for i, batch in enumerate(train_loader):
            subset.append(batch)
            if i >= 3: break
        train_loader = subset

    model = BaselineUNet(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"]).to(device)
    loss_fn = build_loss(cfg, use_boundary=a.use_boundary)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["training"]["amp"] and device.type == "cuda")

    out_dir = Path(a.out or f"outputs/baseline{'_boundary' if a.use_boundary else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    epochs = a.max_epochs or cfg["training"]["baseline_epochs"]
    best_dice = -1.0; history = []
    for epoch in range(1, epochs + 1):
        if a.debug:
            model.train(); total = 0.0
            for batch in train_loader:
                image = batch["image"].to(device); label = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image); loss = loss_fn(logits, label)
                loss.backward(); optimizer.step(); total += loss.item()
            train_loss = total / len(train_loader)
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                device, scaler, cfg["training"]["amp"])
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
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()