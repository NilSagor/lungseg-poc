"""Qualitative overlays: CT | GT | baseline | refined."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference

from lungseg.data import build_val_transforms, load_split
from lungseg.diffusion import DiffusionSchedule
from lungseg.models import BaselineUNet, DiffusionRefiner
from monai.data import Dataset


def overlay(ax, ct, mask, title):
    ax.imshow(ct, cmap="gray")
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
    ax.set_title(title)
    ax.axis("off")


def pad_to_multiple(tensor, multiple=32):
    """
    Pads the spatial dimensions of a tensor to be a multiple of `multiple`.
    Returns the padded tensor and the slices needed to crop it back to the original size.
    """
    spatial_shape = tensor.shape[2:]
    pads = []
    for s in spatial_shape:
        rem = s % multiple
        if rem == 0:
            pads.append((0, 0))
        else:
            pads.append((0, multiple - rem))
            
    # F.pad expects padding amounts starting from the last dimension
    pad_dims = []
    for pad_left, pad_right in reversed(pads):
        pad_dims.extend([pad_left, pad_right])
        
    # Build crop slices in the original dimension order
    crop_slices = [slice(None), slice(None)]  # for batch and channel dims
    for i, s in enumerate(spatial_shape):
        pad_left, pad_right = pads[i]
        if pad_right > 0:
            crop_slices.append(slice(0, s))
        else:
            crop_slices.append(slice(None))
            
    if any(p > 0 for p in pad_dims):
        padded = F.pad(tensor, pad_dims, mode='constant', value=0)
        return padded, crop_slices
    return tensor, crop_slices


def crop_to_original(tensor, crop_slices):
    """Crops the tensor back to its original spatial dimensions."""
    return tensor[crop_slices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--refiner-ckpt", required=True)
    p.add_argument("--out", default="outputs/figures")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_split(cfg["data"]["split_file"])
    ds = Dataset(data=split["val"], transform=build_val_transforms(cfg))

    baseline = BaselineUNet(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"]
    ).to(device)
    baseline.load_state_dict(torch.load(a.baseline_ckpt, map_location=device))
    baseline.eval()

    refiner = DiffusionRefiner(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"]
    ).to(device)
    refiner.load_state_dict(torch.load(a.refiner_ckpt, map_location=device))
    refiner.eval()

    schedule = DiffusionSchedule(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"]
    ).to(device)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roi = tuple(cfg["data"]["patch_size"])

    with torch.no_grad():
        for i in range(min(4, len(ds))):
            sample = ds[i]
            image = sample["image"][None].to(device)
            label = sample["label"][None].to(device)
            
            # 1. Pad image to ensure spatial dimensions are multiples of 32 
            # (required for U-Net skip connections to match without dimension errors)
            image_padded, crop_slices = pad_to_multiple(image, multiple=32)
            
            # 2. Run baseline inference on the padded image
            logits = sliding_window_inference(
                inputs=image_padded, 
                roi_size=roi,
                sw_batch_size=cfg["evaluation"]["sw_batch_size"],
                predictor=baseline, 
                overlap=cfg["evaluation"]["overlap"]
            )
            
            # 3. Run refiner on the padded image and logits
            refined_padded = schedule.sample(
                refiner=refiner, 
                image=image_padded,
                initial_logits=logits,
                inference_steps=cfg["diffusion"]["inference_steps"]
            )
            
            # 4. Crop predictions back to original spatial dimensions
            refined = crop_to_original(refined_padded, crop_slices)
            logits = crop_to_original(logits, crop_slices)

            ct = image[0, 0].cpu().numpy()
            gt = label[0, 0].cpu().numpy()
            b = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(float)
            r = (torch.sigmoid(refined)[0, 0].cpu().numpy() > 0.5).astype(float)

            z = ct.shape[0] // 2
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            overlay(axes[0], ct[z], np.zeros_like(gt[z]), "CT")
            overlay(axes[1], ct[z], gt[z], "GT")
            overlay(axes[2], ct[z], b[z], "Baseline")
            overlay(axes[3], ct[z], r[z], "Refined")
            fig.tight_layout()
            fig.savefig(out / f"case_{i:02d}.png", dpi=150)
            plt.close(fig)

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()