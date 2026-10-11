"""Aggregate per-seed metrics: mean/std/95% CI across seeds, plus paired
differences vs E0 (same seed = pair). If metrics.json contains a "per_case"
list, also a paired bootstrap over cases (averaged over seeds)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXPS = ["e0_unet", "e1_boundary", "e2_diffusion", "e3_diffusion_boundary"]
METRICS = ("dice", "iou", "hd95")
BASE = "e0_unet"


# def load_seed_metrics(root: Path, exp: str, seed: int):
#     f = root / f"{exp}_seed{seed}" / "history.json"
#     return json.loads(f.read_text()) if f.exists() else None


def load_seed_metrics(root: Path, exp: str, seed: int) -> dict | None:
    f = root / f"{exp}_seed{seed}" / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError as e:
        print(f"[warn] {f} is not valid JSON: {e}", file=sys.stderr)
        return None



def t_ci(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, float(m - h), float(m + h)


def paired_case_bootstrap(a, b, n_boot=5000, alpha=0.05, seed=0):
    """a, b: (n_seeds, n_cases). Resample cases; seeds are averaged first."""
    d = a.mean(0) - b.mean(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1010])
    p.add_argument("--out", default="outputs/multiseed_summary.csv")
    p.add_argument("--strict", action="store_true", help="fail if any run is missing")
    p.add_argument("--baseline-exp", default="e0_unet",
                       help="Reference experiment for paired comparisons")
    a = p.parse_args()
    
    
    
    

    root = Path(a.root)
    scal, cases = {}, {}   # [(exp, metric)] -> {seed: value / np.array}
    for exp in EXPS:
        for s in a.seeds:
            m = load_seed_metrics(root, exp, s)
            if m is None:
                print(f"WARNING: missing {exp} seed {s}", file=sys.stderr)
                continue
            for metric in METRICS:
                v = m.get(metric)
                if v is not None and np.isfinite(v):
                    scal.setdefault((exp, metric), {})[s] = float(v)
                pc = m.get("per_case")
                if pc and metric in pc[0]:
                    cases.setdefault((exp, metric), {})[s] = np.array([c[metric] for c in pc], float)

    rows, incomplete = [], False
    for exp in EXPS:
        for metric in METRICS:
            d = scal.get((exp, metric), {})
            mean, lo, hi = t_ci(list(d.values()))
            row = {"experiment": exp, "metric": metric, "n_seeds": len(d),
                   "mean": mean, "ci95_lo": lo, "ci95_hi": hi,
                   "std": float(np.std(list(d.values()), ddof=1)) if len(d) > 1 else float("nan")}
            incomplete |= len(d) != len(a.seeds)

            # if exp != BASE:
            if exp != a.baseline_exp:
                # base = scal.get((BASE, metric), {})
                base = scal.get((a.baseline_exp, metric), {})
                common = sorted(set(d) & set(base))
                dm, dlo, dhi = t_ci([d[s] - base[s] for s in common])
                row.update(delta_vs_e0=dm, delta_ci95_lo=dlo, delta_ci95_hi=dhi, n_pairs=len(common))
                # ce, cb = cases.get((exp, metric), {}), cases.get((BASE, metric), {})
                ce, cb = cases.get((exp, metric), {}), cases.get((a.baseline_exp, metric), {})
                cs = sorted(set(ce) & set(cb))
                if cs and len({len(ce[s]) for s in cs} | {len(cb[s]) for s in cs}) == 1:
                    cm, clo, chi = paired_case_bootstrap(
                        np.stack([ce[s] for s in cs]), np.stack([cb[s] for s in cs]))
                    row.update(case_boot_delta=cm, case_boot_lo=clo, case_boot_hi=chi)
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    if a.strict and incomplete:
        sys.exit("Incomplete: not every (experiment, metric) has all seeds")


if __name__ == "__main__":
    main()