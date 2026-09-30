"""Monte-Carlo uncertainty from the diffusion refiner + calibration metrics."""

from collections.abc import Sequence

import numpy as np
import torch

from lungseg.inference import patch_diffusion_sample


@torch.no_grad()
def sample_masks(
    schedule,
    refiner,
    image,
    initial_logits,
    K: int = 8,
    inference_steps: int | None = None,
    use_patch: bool = True,
    roi_size=(64, 64, 64),
    overlap: float = 0.25,
    base_seed: int = 0,
    store_device: str = "cpu",
) -> torch.Tensor:
    """Return (K, B, 1, D, H, W) stack of sigmoid probabilities.

    Samples are moved to `store_device` as they are produced so K full volumes
    never sit on the GPU at once. Sampling is seeded (base_seed + k) in both
    branches so results are reproducible.
    """
    probs = []
    for k in range(K):
        seed = base_seed + k
        if use_patch:
            logits = patch_diffusion_sample(
                schedule, refiner, image, initial_logits,
                roi_size=roi_size, overlap=overlap,
                inference_steps=inference_steps, seed=seed,
            )
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logits = schedule.sample(
                refiner, image, initial_logits, inference_steps=inference_steps
            )
        probs.append(torch.sigmoid(logits).to(store_device))
    return torch.stack(probs, dim=0)


def predictive_entropy(mean_prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = mean_prob.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def voxel_variance(probs: torch.Tensor) -> torch.Tensor:
    """probs: (K, B, 1, D, H, W) -> variance over K."""
    return probs.var(dim=0, unbiased=False)



def calibration_bins(
    mean_prob: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-bin sufficient statistics (count, sum of confidence, sum of positives).
 
    These are additive, so they can be summed over cases to get a dataset-level
    (voxel-pooled) ECE / reliability diagram. O(N), no Python loop over voxels.
    """
    p = mean_prob.detach().flatten().double().clamp(0, 1)
    y = (target.detach().flatten() > 0).double()
    if mask is not None:
        m = mask.flatten().bool()
        p, y = p[m], y[m]
    idx = torch.clamp((p * n_bins).long(), max=n_bins - 1)
    return {
        "count": torch.bincount(idx, minlength=n_bins).double(),
        "sum_conf": torch.bincount(idx, weights=p, minlength=n_bins),
        "sum_acc": torch.bincount(idx, weights=y, minlength=n_bins),
    }

def ece_from_bins(count, sum_conf, sum_acc) -> float:
    count = torch.as_tensor(count, dtype=torch.float64)
    sum_conf = torch.as_tensor(sum_conf, dtype=torch.float64)
    sum_acc = torch.as_tensor(sum_acc, dtype=torch.float64)
    n = count.sum()
    if n == 0:
        return float("nan")
    nz = count > 0
    gap = (sum_conf[nz] - sum_acc[nz]).abs() / count[nz]
    return float(((count[nz] / n) * gap).sum())


def expected_calibration_error(
    mean_prob: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    mask: torch.Tensor | None = None,
) -> float:
    """ECE for binary segmentation over voxels (optionally within `mask`)."""
    return ece_from_bins(**calibration_bins(mean_prob, target, n_bins, mask))







# ----------------------------------------------------------------------
# Selective prediction
# ----------------------------------------------------------------------

def dice_score(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5,
               eps: float = 1e-6) -> torch.Tensor:
    """Per-volume Dice, shape (B,). Empty-vs-empty counts as 1.0."""
    pred = (prob > thr).float()
    tgt = (target > 0).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * tgt).sum(dim=dims)
    denom = pred.sum(dim=dims) + tgt.sum(dim=dims)
    dice = 2 * inter / denom.clamp_min(eps)
    return torch.where(denom == 0, torch.ones_like(dice), dice)


def volume_confidence(mean_prob: torch.Tensor, region_thr: float = 0.05) -> float:
    """Scalar confidence for one volume = -mean entropy over the candidate region.

    Averaging over the whole volume is dominated by easy background, so we only
    average over voxels with mean_prob > region_thr (falls back to all voxels).
    """
    ent = predictive_entropy(mean_prob)
    region = mean_prob > region_thr
    val = ent[region].mean() if region.any() else ent.mean()
    return float(-val.item())

def risk_coverage_curve(
    dice: Sequence[float],
    confidence: Sequence[float],
    coverages: Sequence[float] | None = None,
) -> dict[str, object]:
    """Dataset-level selective prediction: keep the top-`coverage` most
    confident volumes and report mean Dice on them.

    Must be called ONCE with arrays over all cases (not per case).
    Also returns an oracle curve (ranking by true Dice) and AURC
    (area under risk-coverage, risk = 1 - Dice; lower is better).
    """
    dice = np.asarray(dice, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    n = len(dice)
    if n == 0:
        return {"coverage": [], "dice": [], "oracle_dice": [], "aurc": float("nan")}
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 10)

    d_sorted = dice[np.argsort(-conf, kind="stable")]
    oracle = np.sort(dice)[::-1]
    cum_risk = np.cumsum(1.0 - d_sorted) / np.arange(1, n + 1)

    cov_out, dice_out, oracle_out = [], [], []
    for c in coverages:
        k = max(1, int(np.ceil(c * n)))
        cov_out.append(k / n)
        dice_out.append(float(d_sorted[:k].mean()))
        oracle_out.append(float(oracle[:k].mean()))
    return {"coverage": cov_out, "dice": dice_out,
            "oracle_dice": oracle_out, "aurc": float(cum_risk.mean())}


def uncertainty_error_correlation(
    mean_prob: torch.Tensor,
    target: torch.Tensor,
    max_voxels: int = 2_000_000,
    seed: int = 0,
) -> dict[str, float]:
    """Pearson(entropy, error) and AUROC of entropy for detecting errors,
    on a fixed random voxel subsample (full volumes are ~1e7-1e8 voxels)."""
    ent = predictive_entropy(mean_prob).flatten().cpu().numpy()
    err = ((mean_prob > 0.5) != (target > 0)).flatten().cpu().numpy().astype(np.float32)

    if ent.size > max_voxels:
        idx = np.random.default_rng(seed).choice(ent.size, max_voxels, replace=False)
        ent, err = ent[idx], err[idx]

    if err.std() == 0 or ent.std() == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(ent, err)[0, 1])

    n_pos = int(err.sum())
    n_neg = err.size - n_pos
    if n_pos == 0 or n_neg == 0:
        auroc = float("nan")
    else:
        from scipy.stats import rankdata
        ranks = rankdata(ent)  # average ranks for ties
        auroc = float((ranks[err == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"pearson_entropy_error": pearson, "auroc_entropy_error": auroc}



