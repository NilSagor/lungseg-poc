#!/usr/bin/env bash
# Multi-seed runner: 5 seeds x E0-E3. Re-runnable: a run counts as finished only
# when its metrics.json exists (set FORCE=1 to redo everything).
set -euo pipefail

CFG=${CFG:-configs/poc.yaml}
read -r -a SEEDS <<< "${SEEDS:-42 123 456 789 1010}"
FORCE=${FORCE:-0}

done_already() { [ "$FORCE" != "1" ] && [ -f "$1" ]; }

for SEED in "${SEEDS[@]}"; do
  echo "=== seed=$SEED ==="

  for EXP in e0_unet e1_boundary; do
    OUT=outputs/${EXP}_seed${SEED}
    if done_already "$OUT/metrics.json"; then echo "skip $OUT"; continue; fi
    EXTRA=()
    if [ "$EXP" = "e1_boundary" ]; then EXTRA+=(--use-boundary); fi
    python scripts/train_baseline.py --config "$CFG" --out "$OUT" \
      --seed-override "$SEED" ${EXTRA[@]+"${EXTRA[@]}"}
  done

  OUT=outputs/e2_diffusion_seed${SEED}
  if done_already "$OUT/metrics.json"; then echo "skip $OUT"; else
    python scripts/train_diffusion.py --config "$CFG" \
      --baseline-ckpt outputs/e0_unet_seed${SEED}/best.pt \
      --out "$OUT" --seed-override "$SEED"
  fi

  OUT=outputs/e3_diffusion_boundary_seed${SEED}
  if done_already "$OUT/metrics.json"; then echo "skip $OUT"; else
    python scripts/train_diffusion.py --config "$CFG" \
      --baseline-ckpt outputs/e1_boundary_seed${SEED}/best.pt \
      --use-boundary --out "$OUT" --seed-override "$SEED"
  fi
done