"""Reliability diagram, risk-coverage curves, entropy-vs-error scatter."""

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reliability(ax, b, label):
    cnt = np.asarray(b["count"]); ok = cnt > 0
    conf = np.asarray(b["sum_conf"])[ok] / cnt[ok]
    acc = np.asarray(b["sum_acc"])[ok] / cnt[ok]
    ax.plot(conf, acc, marker="o", label=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out", default="outputs/figures/calibration.png")
    a = p.parse_args()
    s = json.loads(Path(a.summary).read_text())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    reliability(ax, s["calibration_bins"]["diffusion_roi"], f"diffusion (ECE-ROI {s['ece']['diffusion_roi']:.3f})")
    reliability(ax, s["calibration_bins"]["baseline_roi"], f"baseline (ECE-ROI {s['ece']['baseline_roi']:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction positive")
    ax.set_title("Reliability (voxels near tumour)"); ax.legend(); ax.grid(True)

    ax = axes[1]
    rc = s["risk_coverage"]
    ax.plot(rc["diffusion"]["coverage"], rc["diffusion"]["dice"], marker="o",
            label=f"diffusion (AURC {rc['diffusion']['aurc']:.3f})")
    ax.plot(rc["baseline"]["coverage"], rc["baseline"]["dice"], marker="s",
            label=f"baseline (AURC {rc['baseline']['aurc']:.3f})")
    ax.plot(rc["diffusion"]["coverage"], rc["diffusion"]["oracle_dice"], "k--", label="oracle")
    ax.set_xlabel("Coverage (fraction of volumes kept)"); ax.set_ylabel("Mean Dice on retained")
    ax.set_title("Risk-coverage"); ax.legend(); ax.grid(True)

    ax = axes[2]
    ax.scatter([c["mean_entropy"] for c in s["per_case"]],
               [c["auroc_entropy_error"] for c in s["per_case"]])
    ax.axhline(0.5, color="k", ls="--")
    ax.set_xlabel("Mean predictive entropy"); ax.set_ylabel("AUROC (entropy -> voxel error)")
    ax.set_title("Does entropy flag errors?"); ax.grid(True)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()