"""Evaluate a saved baseline / refiner checkpoint on the validation split."""
import argparse, json
from pathlib import Path
import torch
import yaml
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from lungseg.data import build_dataloaders
from lungseg.diffusion import DiffusionSchedule
from lungseg.metrics import SegmentationMetrics
from lungseg.models import BaselineUNet, DiffusionRefiner


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--refiner-ckpt", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])

    baseline = BaselineUNet(channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"]).to(device)
    baseline.load_state_dict(torch.load(a.baseline_ckpt, map_location=device))
    baseline.eval()

    refiner, schedule = None, None
    if a.refiner_ckpt:
        refiner = DiffusionRefiner(channels=tuple(cfg["model"]["channels"]),
            num_res_units=cfg["model"]["num_res_units"]).to(device)
        refiner.load_state_dict(torch.load(a.refiner_ckpt, map_location=device))
        refiner.eval()
        schedule = DiffusionSchedule(timesteps=cfg["diffusion"]["timesteps"],
            beta_start=cfg["diffusion"]["beta_start"],
            beta_end=cfg["diffusion"]["beta_end"]).to(device)

    metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
    roi = tuple(cfg["data"]["patch_size"])
    for batch in tqdm(val_loader, desc="eval"):
        image = batch["image"].to(device); label = batch["label"].to(device)
        logits = sliding_window_inference(inputs=image, roi_size=roi,
            sw_batch_size=cfg["evaluation"]["sw_batch_size"],
            predictor=baseline, overlap=cfg["evaluation"]["overlap"])
        if refiner is not None:
            logits = schedule.sample(refiner=refiner, image=image,
                initial_logits=logits,
                inference_steps=cfg["diffusion"]["inference_steps"])
        metrics.update(logits, label)

    result = metrics.aggregate()
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()