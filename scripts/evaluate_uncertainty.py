"""Generate MC samples, uncertainty maps, calibration metrics for one checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import Dataset
from monai.inferers import sliding_window_inference

from lungseg.data import build_val_transforms, load_split
from lungseg.diffusion import DiffusionSchedule
from lungseg.evaluation import (
    calibration_bins,
    dice_score,
    ece_from_bins,
    predictive_entropy,
    risk_coverage_curve,
    sample_masks,
    uncertainty_error_correlation,
    volume_confidence,
    voxel_variance,
)
from lungseg.models import BaselineUNet, DiffusionRefiner


def _add(acc, bins):
    if acc is None:
        return {k: v.clone() for k, v in bins.items()}
    for k in acc:
        acc[k] += bins[k]
    return acc


def _as_lists(bins):
    return {k: v.tolist() for k, v in bins.items()}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--refiner-ckpt", required=True)
    p.add_argument("--K", type=int, default=None, help="default: cfg.uncertainty.K")
    p.add_argument("--n-bins", type=int, default=None, help="default: cfg.uncertainty.n_bins_ece")
    p.add_argument("--save-tensors", action="store_true",
                   help="save fp16 mean/variance/entropy maps per case (large)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    K = a.K or cfg.get("uncertainty", {}).get("K", 8)
    n_bins = a.n_bins or cfg.get("uncertainty", {}).get("n_bins_ece", 15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))

    split = load_split(cfg["data"]["split_file"])
    ds = Dataset(data=split["val"], transform=build_val_transforms(cfg))
    roi = tuple(cfg["data"]["patch_size"])

    baseline = BaselineUNet(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device)
    baseline.load_state_dict(torch.load(a.baseline_ckpt, map_location=device))
    baseline.eval()

    refiner = DiffusionRefiner(
        channels=tuple(cfg["model"]["channels"]),
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device)
    refiner.load_state_dict(torch.load(a.refiner_ckpt, map_location=device))
    refiner.eval()

    schedule = DiffusionSchedule(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    if a.save_tensors:
        (out_dir / "per_case").mkdir(exist_ok=True)

    use_patch = cfg["evaluation"].get("diffusion_sampler", "full") == "patch"
    per_case = []
    bins = {"diffusion": None, "diffusion_roi": None, "baseline": None, "baseline_roi": None}
    dice_d, dice_b, conf_d, conf_b = [], [], [], []

    for i in range(len(ds)):
        sample = ds[i]
        image = sample["image"][None].to(device)
        label = (sample["label"][None] > 0).float()  # CPU, binary

        initial_logits = sliding_window_inference(
            inputs=image, roi_size=roi,
            sw_batch_size=cfg["evaluation"]["sw_batch_size"],
            predictor=baseline, overlap=cfg["evaluation"]["overlap"],
        )
        base_p = torch.sigmoid(initial_logits).cpu()

        probs = sample_masks(
            schedule, refiner, image, initial_logits,
            K=K,
            inference_steps=cfg["diffusion"]["inference_steps"],
            use_patch=use_patch,
            roi_size=roi,
            overlap=cfg["evaluation"]["overlap"],
            base_seed=1000 * i,
            store_device="cpu",
        )  # (K,1,1,D,H,W) on CPU
        mean_p = probs.mean(0)
        var = voxel_variance(probs)
        ent = predictive_entropy(mean_p)

        # Background voxels dominate ECE; also report it near the tumour.
        for name, prob in (("diffusion", mean_p), ("baseline", base_p)):
            roi_mask = (prob > 0.01) | (label > 0)
            bins[name] = _add(bins[name], calibration_bins(prob, label, n_bins))
            bins[name + "_roi"] = _add(bins[name + "_roi"], calibration_bins(prob, label, n_bins, roi_mask))

        d_d = float(dice_score(mean_p, label)[0]); d_b = float(dice_score(base_p, label)[0])
        c_d = volume_confidence(mean_p); c_b = volume_confidence(base_p)
        dice_d.append(d_d); dice_b.append(d_b); conf_d.append(c_d); conf_b.append(c_b)

        corr = uncertainty_error_correlation(mean_p, label)
        if a.save_tensors:
            torch.save({
                "mean_prob": mean_p.half(), "variance": var.half(),
                "entropy": ent.half(), "label": label.to(torch.uint8),
            }, out_dir / "per_case" / f"case_{i:03d}.pt")

        per_case.append({
            "case": i, "dice_mc_mean": d_d, "dice_baseline": d_b,
            "confidence": c_d, "mean_entropy": float(ent.mean()),
            "mean_variance": float(var.mean()), **corr,
        })
        print(f"[{i + 1}/{len(ds)}] dice base={d_b:.3f} mc={d_d:.3f}", flush=True)

    summary = {
        "K": K, "n_bins": n_bins, "n_cases": len(per_case),
        "ece": {k: ece_from_bins(**v) for k, v in bins.items()},
        "calibration_bins": {k: _as_lists(v) for k, v in bins.items()},
        "risk_coverage": {
            "diffusion": risk_coverage_curve(dice_d, conf_d),
            "baseline": risk_coverage_curve(dice_b, conf_b),
        },
        "mean_pearson_entropy_error": float(np.mean([c["pearson_entropy_error"] for c in per_case])),
        "mean_auroc_entropy_error": float(np.nanmean([c["auroc_entropy_error"] for c in per_case])),
        "per_case": per_case,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("K", "n_cases", "ece", "mean_pearson_entropy_error",
                       "mean_auroc_entropy_error")}, indent=2))
    print("AURC diffusion / baseline:",
          summary["risk_coverage"]["diffusion"]["aurc"],
          summary["risk_coverage"]["baseline"]["aurc"])


if __name__ == "__main__":
    main()