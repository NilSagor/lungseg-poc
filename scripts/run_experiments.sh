#!/usr/bin/env bash
set -euo pipefail
CFG=configs/poc.yaml

echo "=== E0: baseline ==="
python scripts/train_baseline.py --config $CFG --out outputs/e0_unet
python scripts/evaluate.py --config $CFG \
    --baseline-ckpt outputs/e0_unet/best.pt \
    --out outputs/e0_unet/metrics.json

echo "=== E1: baseline + boundary ==="
python scripts/train_baseline.py --config $CFG --use-boundary --out outputs/e1_boundary
python scripts/evaluate.py --config $CFG \
    --baseline-ckpt outputs/e1_boundary/best.pt \
    --out outputs/e1_boundary/metrics.json

echo "=== E2: baseline + diffusion ==="
python scripts/train_diffusion.py --config $CFG \
    --baseline-ckpt outputs/e0_unet/best.pt \
    --out outputs/e2_diffusion
python scripts/evaluate.py --config $CFG \
    --baseline-ckpt outputs/e0_unet/best.pt \
    --refiner-ckpt outputs/e2_diffusion/last.pt \
    --out outputs/e2_diffusion/metrics.json

echo "=== E3: baseline + diffusion + boundary ==="
python scripts/train_diffusion.py --config $CFG \
    --baseline-ckpt outputs/e1_boundary/best.pt \
    --use-boundary \
    --out outputs/e3_diffusion_boundary
python scripts/evaluate.py --config $CFG \
    --baseline-ckpt outputs/e1_boundary/best.pt \
    --refiner-ckpt outputs/e3_diffusion_boundary/last.pt \
    --out outputs/e3_diffusion_boundary/metrics.json

echo "All experiments complete."