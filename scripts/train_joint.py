"""Joint baseline + diffusion refiner training (Phase 8)."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lungseg.data import build_dataloaders
from lungseg.diffusion import DiffusionSchedule
from lungseg.inference import patch_diffusion_sample
from lungseg.metrics import SegmentationMetrics
from lungseg.models import BaselineUNet, DiffusionRefiner
from lungseg.training.joint_trainer import JointTrainer


def maybe_load(module, ckpt, device):
    if ckpt and Path(ckpt).exists():
        module.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded weights from {ckpt}")
    return module


@torch.no_grad()
def validate(baseline, refiner, schedule, loader, cfg, device, use_patch):
    baseline.eval(); refiner.eval()
    metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
    roi = tuple(cfg["data"]["patch_size"])
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = sliding_window_inference(
            inputs=image, roi_size=roi,
            sw_batch_size=cfg["evaluation"]["sw_batch_size"],
            predictor=baseline, overlap=cfg["evaluation"]["overlap"],
        )
        if use_patch:
            
            logits = patch_diffusion_sample(
                schedule, refiner, image, logits,
                roi_size=roi, overlap=cfg["evaluation"]["overlap"],
                inference_steps=cfg["diffusion"]["inference_steps"],
                seed=cfg["seed"],
            )
        else:
            logits = schedule.sample(
                refiner, image, logits,
                inference_steps=cfg["diffusion"]["inference_steps"],
            )
        metrics.update(logits, label)
    return metrics.aggregate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-ckpt", default=None)
    p.add_argument("--refiner-ckpt", default=None)
    p.add_argument("--mode", choices=["warmup", "alternating", "end_to_end"],
                   default="alternating")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])
    baseline = BaselineUNet(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device)
    refiner = DiffusionRefiner(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device)

    baseline = maybe_load(baseline, a.baseline_ckpt, device)
    refiner = maybe_load(refiner, a.refiner_ckpt, device)

    schedule = DiffusionSchedule(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    trainer = JointTrainer(baseline, refiner, schedule, cfg, device)

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    use_patch = cfg["evaluation"].get("diffusion_sampler", "full") == "patch"
    history = []
    for epoch in range(1, cfg["training"]["diffusion_epochs"] + 1):
        loss = trainer.train_one_epoch(train_loader, mode=a.mode)
        writer.add_scalar(f"joint/{a.mode}/loss", loss, epoch)
        print(f"epoch {epoch:03d} | mode={a.mode} | loss={loss:.4f}")

        if epoch % cfg["training"]["val_interval"] == 0:
            val = validate(baseline, refiner, schedule, val_loader, cfg, device, use_patch)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(f"          val: {val}")
            history.append({"epoch": epoch, "mode": a.mode, "loss": loss, **val})

    torch.save(baseline.state_dict(), out_dir / "baseline.pt")
    torch.save(refiner.state_dict(), out_dir / "refiner.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved to {out_dir}")

    if history:
        final_metrics = validate(
            baseline, refiner, schedule, val_loader, cfg, device, use_patch
        )
        final_metrics["epoch"] = history[-1]["epoch"]
        final_metrics["seed"] = cfg["seed"]
        final_metrics["mode"] = a.mode
        final_metrics["sampler"] = cfg["evaluation"].get("diffusion_sampler", "full")
        (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
        print(f"Wrote {out_dir / 'metrics.json'}: {final_metrics}")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()