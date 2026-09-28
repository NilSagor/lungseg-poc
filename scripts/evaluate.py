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

import torch.nn.functional as F

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Pads the spatial dimensions of a tensor to be divisible by `multiple`."""
    spatial_shape = tensor.shape[2:]
    pad_sizes = []
    # F.pad expects padding starting from the last dimension
    for s in reversed(spatial_shape):
        remainder = s % multiple
        pad = (multiple - remainder) % multiple
        pad_sizes.extend([0, pad])
    
    if sum(pad_sizes) > 0:
        tensor = F.pad(tensor, pad_sizes, mode='constant', value=0)
    return tensor

def crop_to_original(tensor: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Crops the spatial dimensions of a tensor back to the original shape."""
    slices = [slice(None), slice(None)]  # Keep batch and channel dimensions intact
    for s in original_shape[2:]:
        slices.append(slice(0, s))
    return tensor[tuple(slices)]


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
    pad_multiple = 16
    for batch in tqdm(val_loader, desc="eval"):
        image = batch["image"].to(device) 
        label = batch["label"].to(device)
        logits = sliding_window_inference(
            inputs=image, 
            roi_size=roi,
            sw_batch_size=cfg["evaluation"]["sw_batch_size"],
            predictor=baseline, 
            overlap=cfg["evaluation"]["overlap"]
        )
        if refiner is not None:
            padded_image = pad_to_multiple(image, pad_multiple)
            padded_logits = pad_to_multiple(logits, pad_multiple)

            refined_logits = schedule.sample(
                refiner=refiner, 
                image=padded_image,
                initial_logits=padded_logits,
                inference_steps=cfg["diffusion"]["inference_steps"]
            )
            
            logits = crop_to_original(refined_logits, image.shape)
            # logits = schedule.sample(refiner=refiner, image=image,
            #     initial_logits=logits,
            #     inference_steps=cfg["diffusion"]["inference_steps"])
        metrics.update(logits, label)

    result = metrics.aggregate()
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()