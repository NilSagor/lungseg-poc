"""Train a 3D segmentation baseline."""
"""Train a 3D segmentation baseline (E0 / E1 and the Phase 11-14 baselines) with Lightning.

Same command line and the same outputs as before: best.pt / last.pt (plain
state_dicts), history.json, metrics.json, tb/. Lightning adds resumable
checkpoints in <out>/ckpt/ (best.ckpt, last.ckpt) and the --resume flag.
"""
import argparse
import json
import math
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from lungseg.config import load_config
from lungseg.data import build_dataloaders
from lungseg.training.lightning_module import HistoryRecorder, LitSegmentation, PlainWeightsExport


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--debug", action="store_true",
                   help="overfit check: 4 cached training batches, no validation")
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--use-boundary", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed-override", type=int, default=42)
    p.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience in validation checks (0 = off); "
                        "overrides training.early_stopping_patience")
    p.add_argument("--resume", default=None, help="Lightning checkpoint (.ckpt) to resume from")
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.seed_override is not None:
        cfg["seed"] = a.seed_override
    L.seed_everything(cfg["seed"], workers=True)
    set_determinism(cfg["seed"])  # MONAI transforms, as before

    train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])
    if a.debug:
        cached = [b for _, b in zip(range(4), train_loader)]
        train_loader = DataLoader(cached, batch_size=None)  # same 4 batches every epoch
        val_loader = None

    module = LitSegmentation(cfg, use_boundary=a.use_boundary)
    n_params = sum(p.numel() for p in module.model.parameters())
    model_name = cfg["model"].get("name", "unet3d")
    print(f"Model: {model_name} ({n_params / 1e6:.2f}M params)")

    out_dir = Path(a.out or f"outputs/baseline{'_boundary' if a.use_boundary else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = cfg["training"]
    epochs = a.max_epochs or t["baseline_epochs"]
    patience = a.patience if a.patience is not None else t.get("early_stopping_patience", 0)

    prior = []  # when resuming, keep the epochs already recorded and the best Dice so far
    if a.resume and (out_dir / "history.json").exists():
        prior = json.loads((out_dir / "history.json").read_text())
    history = HistoryRecorder(prior)
    best_so_far = max((r["dice"] for r in prior), default=-math.inf)
    callbacks = [history, PlainWeightsExport(out_dir, best=best_so_far)]
    stopper = None
    if patience and not a.debug:
        stopper = EarlyStopping(monitor="val/dice", mode="max", patience=patience,
                                min_delta=t.get("early_stopping_min_delta", 0.0))
        callbacks.append(stopper)
    if not a.debug:
        callbacks.append(ModelCheckpoint(
            dirpath=out_dir / "ckpt", filename="best", monitor="val/dice", mode="max",
            save_top_k=1, save_last=True))

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if t["amp"] and torch.cuda.is_available() else 32,
        check_val_every_n_epoch=t["val_interval"],
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(str(out_dir), name="tb", version=""),
        callbacks=callbacks,
        enable_checkpointing=not a.debug,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=a.resume)

    (out_dir / "history.json").write_text(json.dumps(history.rows, indent=2))
    if history.rows and not a.debug:
        best = max(history.rows, key=lambda r: r["dice"])  # same rule as best.pt
        (out_dir / "metrics.json").write_text(json.dumps({
            **best,
            "seed": cfg["seed"],
            "use_boundary": bool(a.use_boundary),
            "model": model_name,
            "n_params": n_params,
            "checkpoint": "best.pt",
            "epochs_run": history.rows[-1]["epoch"],
            "stopped_early": bool(stopper and stopper.stopped_epoch > 0),
        }, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()












# import argparse
# import json
# import random
# from pathlib import Path

# import numpy as np
# import torch
# from monai.inferers import sliding_window_inference
# from monai.utils import set_determinism
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from lungseg.config import load_config
# from lungseg.data import build_dataloaders
# from lungseg.losses import BCEDiceLoss, BoundaryLoss, deep_supervision_loss
# from lungseg.metrics import SegmentationMetrics
# from lungseg.models import build_from_config


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     set_determinism(seed)


# def build_loss(cfg, use_boundary):
#     seg = BCEDiceLoss()
#     bnd = BoundaryLoss() if use_boundary else None
#     w = cfg["training"]["boundary_weight"]

#     def loss_fn(logits, target):
#         loss = seg(logits, target)
#         if bnd is not None:
#             loss = loss + w * bnd(logits, target)
#         return loss
#     return loss_fn


# def step(model, batch, optimizer, loss_fn, device, scaler=None, use_amp=False):
#     image = batch["image"].to(device, non_blocking=True)
#     label = batch["label"].to(device, non_blocking=True)
#     optimizer.zero_grad(set_to_none=True)
#     if use_amp:
#         with torch.autocast(device_type="cuda", dtype=torch.float16):
#             loss = deep_supervision_loss(loss_fn, model(image), label)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#     else:
#         loss = deep_supervision_loss(loss_fn, model(image), label)
#         loss.backward()
#         optimizer.step()
#     return loss.item()


# def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp, show_progress=True):
#     model.train()
#     running = 0.0
#     it = tqdm(loader, desc="train", leave=False) if show_progress else loader
#     for batch in it:
#         running += step(model, batch, optimizer, loss_fn, device, scaler, use_amp)
#     return running / max(len(loader), 1)


# @torch.no_grad()
# def validate(model, loader, cfg, device):
#     model.eval()
#     metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
#     roi = tuple(cfg["data"]["patch_size"])
#     for batch in tqdm(loader, desc="val", leave=False):
#         image = batch["image"].to(device)
#         label = batch["label"].to(device)
#         logits = sliding_window_inference(
#             inputs=image, roi_size=roi,
#             sw_batch_size=cfg["evaluation"]["sw_batch_size"],
#             predictor=model, overlap=cfg["evaluation"]["overlap"])
#         metrics.update(logits, label)
#     return metrics.aggregate()


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--config", required=True)
#     p.add_argument("--debug", action="store_true")
#     p.add_argument("--max_epochs", type=int, default=None)
#     p.add_argument("--use-boundary", action="store_true")
#     p.add_argument("--out", default=None)
#     p.add_argument("--seed-override", type=int, default=42)
#     a = p.parse_args()

#     cfg = load_config(a.config)
#     if a.seed_override is not None:
#         cfg["seed"] = a.seed_override
#     set_seed(cfg["seed"])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])

#     if a.debug:
#         subset = []
#         for i, batch in enumerate(train_loader):
#             subset.append(batch)
#             if i >= 3:
#                 break
#         train_loader = subset

#     model = build_from_config(cfg).to(device)
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Model: {cfg['model'].get('name', 'unet3d')} ({n_params / 1e6:.2f}M params)")

#     loss_fn = build_loss(cfg, use_boundary=a.use_boundary)
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=cfg["training"]["learning_rate"],
#         weight_decay=cfg["training"]["weight_decay"])
#     scaler = torch.amp.GradScaler(
#         enabled=cfg["training"]["amp"] and device.type == "cuda")

#     out_dir = Path(a.out or f"outputs/baseline{'_boundary' if a.use_boundary else ''}")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(out_dir / "tb"))

#     epochs = a.max_epochs or cfg["training"]["baseline_epochs"]
#     best_dice = -1.0
#     history = []
#     for epoch in range(1, epochs + 1):
#         if a.debug:
#             train_loss = train_one_epoch(
#                 model, train_loader, optimizer, loss_fn, device, None, False,
#                 show_progress=False)
#         else:
#             train_loss = train_one_epoch(
#                 model, train_loader, optimizer, loss_fn, device, scaler,
#                 cfg["training"]["amp"])
#         writer.add_scalar("loss/train", train_loss, epoch)
#         print(f"epoch {epoch:03d} | train_loss={train_loss:.4f}")

#         if not a.debug and epoch % cfg["training"]["val_interval"] == 0:
#             val_metrics = validate(model, val_loader, cfg, device)
#             for k, v in val_metrics.items():
#                 writer.add_scalar(f"val/{k}", v, epoch)
#             print(f"          val: {val_metrics}")
#             history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
#             if val_metrics["dice"] > best_dice:
#                 best_dice = val_metrics["dice"]
#                 torch.save(model.state_dict(), out_dir / "best.pt")

#     torch.save(model.state_dict(), out_dir / "last.pt")
#     (out_dir / "history.json").write_text(json.dumps(history, indent=2))

#     if not a.debug and history:
#         best = max(history, key=lambda r: r["dice"])
#         (out_dir / "metrics.json").write_text(json.dumps({
#             **best,
#             "seed": cfg["seed"],
#             "use_boundary": bool(a.use_boundary),
#             "model": cfg["model"].get("name", "unet3d"),
#             "n_params": n_params,
#             "checkpoint": "best.pt",
#         }, indent=2))
#     print(f"Saved to {out_dir}")


# if __name__ == "__main__":
#     main()