"""Compare full-volume vs patch-based diffusion sampling for runtime & peak GPU memory."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from lungseg.diffusion import DiffusionSchedule
from lungseg.inference import patch_diffusion_sample
from lungseg.models import DiffusionRefiner


def measure(fn, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    )
    return out, dt, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--volume-shape", nargs=3, type=int, default=[160, 160, 160])
    p.add_argument("--out", default="outputs/benchmark_sampling/benchmark_sampling.json")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = DiffusionSchedule(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)
    refiner = DiffusionRefiner(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device).eval()

    D, H, W = a.volume_shape
    image = torch.randn(1, 1, D, H, W, device=device)
    logits = torch.zeros_like(image)

    with torch.no_grad():
        _, t_full, m_full = measure(
            lambda: schedule.sample(
                refiner, image, logits,
                inference_steps=cfg["diffusion"]["inference_steps"],
            ),
            device,
        )
        _, t_patch, m_patch = measure(
            lambda: patch_diffusion_sample(
                schedule, refiner, image, logits,
                roi_size=tuple(cfg["data"]["patch_size"]),
                overlap=cfg["evaluation"]["overlap"],
                inference_steps=cfg["diffusion"]["inference_steps"],
            ),
            device,
        )

    result = {
        "volume_shape": [D, H, W],
        "full":  {"seconds": t_full,  "peak_gpu_mb": m_full},
        "patch": {"seconds": t_patch, "peak_gpu_mb": m_patch},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()