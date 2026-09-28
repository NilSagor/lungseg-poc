"""Train the conditional diffusion refiner (E2 / E3) with a frozen baseline."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lungseg.data import build_dataloaders
from lungseg.diffusion import DiffusionSchedule
from lungseg.metrics import SegmentationMetrics
from lungseg.models import BaselineUNet, DiffusionRefiner


def load_baseline(cfg, device, ckpt_path):
    model = BaselineUNet(channels=tuple(cfg["model"]["channels"]),
                         num_res_units=cfg["model"]["num_res_units"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model



def pad_to_multiple(x, divisor):
    """Pad spatial dimensions to be divisible by `divisor` using replicate padding."""
    spatial_dims = x.ndim - 2
    pads = []
    # Calculate padding for each spatial dimension (W, H, D...)
    for i in range(spatial_dims):
        dim_size = x.shape[-(i+1)]
        remainder = dim_size % divisor
        pad_amount = (divisor - remainder) % divisor
        pads.extend([0, pad_amount]) # [pad_left, pad_right] for the current dim
    
    if sum(pads) > 0:
        # 'replicate' mode avoids sharp edge artifacts that 'constant' (0) padding might introduce
        x = F.pad(x, pads, mode="replicate")
    return x

def crop_to_shape(x, original_shape):
    """Crop the padded tensor back to the original spatial dimensions."""
    spatial_shape = original_shape[2:] # Ignore Batch and Channel dimensions
    slices = (slice(None), slice(None)) + tuple(slice(0, s) for s in spatial_shape)
    return x[slices]


def train_one_epoch(refiner, baseline, loader, optimizer, schedule, device, scaler, cfg):
    refiner.train()
    w_diff = cfg["training"]["diffusion_weight"]
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device).float()
        with torch.no_grad():
            initial_logits = baseline(image)
        noise = torch.randn_like(label)
        t = torch.randint(0, schedule.timesteps, (image.shape[0],), device=device)
        noisy_mask = schedule.add_noise(label, noise, t)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=cfg["training"]["amp"]):
            eps_hat = refiner(image, initial_logits, noisy_mask, t)
            loss = w_diff * F.mse_loss(eps_hat, noise)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


# @torch.no_grad()
# def validate(refiner, baseline, loader, schedule, cfg, device):
#     refiner.eval()
#     metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
#     roi = tuple(cfg["data"]["patch_size"])
#     from monai.inferers import sliding_window_inference
#     for batch in tqdm(loader, desc="val", leave=False):
#         image = batch["image"].to(device); label = batch["label"].to(device)
#         initial_logits = sliding_window_inference(
#             inputs=image, roi_size=roi,
#             sw_batch_size=cfg["evaluation"]["sw_batch_size"],
#             predictor=baseline, overlap=cfg["evaluation"]["overlap"])
#         refined = schedule.sample(refiner=refiner, image=image,
#             initial_logits=initial_logits,
#             inference_steps=cfg["diffusion"]["inference_steps"])
#         metrics.update(refined, label)
#     return metrics.aggregate()


@torch.no_grad()
def validate(refiner, baseline, loader, schedule, cfg, device):
    refiner.eval()
    metrics = SegmentationMetrics(threshold=cfg["evaluation"]["threshold"])
    roi = tuple(cfg["data"]["patch_size"])
    from monai.inferers import sliding_window_inference
    
    # Determine the required divisor based on the number of downsampling layers.
    # A standard UNet has len(channels) - 1 downsampling steps.
    # If channels=(16, 32, 64, 128, 256), there are 4 downsampling steps (divisor=16).
    divisor = 2 ** (len(cfg["model"]["channels"]) - 1)
    
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        
        initial_logits = sliding_window_inference(
            inputs=image, roi_size=roi,
            sw_batch_size=cfg["evaluation"]["sw_batch_size"],
            predictor=baseline, overlap=cfg["evaluation"]["overlap"])
            
        original_shape = image.shape
        
        # 1. Pad image and logits so spatial dims are divisible by 2^(num_levels)
        image_padded = pad_to_multiple(image, divisor)
        initial_logits_padded = pad_to_multiple(initial_logits, divisor)
        
        # 2. Run the diffusion refiner on the padded volume
        refined = schedule.sample(refiner=refiner, image=image_padded,
            initial_logits=initial_logits_padded,
            inference_steps=cfg["diffusion"]["inference_steps"])
            
        # 3. Crop the refined mask back to the original image shape
        refined = crop_to_shape(refined, original_shape)
        
        metrics.update(refined, label)
        
    return metrics.aggregate()



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--use-boundary", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg, cfg["data"]["split_file"])
    baseline = load_baseline(cfg, device, a.baseline_ckpt)

    refiner = DiffusionRefiner(channels=tuple(cfg["model"]["channels"]),
                               num_res_units=cfg["model"]["num_res_units"]).to(device)
    schedule = DiffusionSchedule(timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"]).to(device)

    optimizer = torch.optim.AdamW(refiner.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["training"]["amp"] and device.type == "cuda")

    out_dir = Path(a.out or f"outputs/diffusion{'_boundary' if a.use_boundary else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    history = []
    for epoch in range(1, cfg["training"]["diffusion_epochs"] + 1):
        train_loss = train_one_epoch(refiner, baseline, train_loader, optimizer,
            schedule, device, scaler, cfg)
        writer.add_scalar("loss/train", train_loss, epoch)
        print(f"epoch {epoch:03d} | diff_loss={train_loss:.4f}")
        if epoch % cfg["training"]["val_interval"] == 0:
            val_metrics = validate(refiner, baseline, val_loader, schedule, cfg, device)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(f"          val: {val_metrics}")
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
    torch.save(refiner.state_dict(), out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()