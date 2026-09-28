"""Patch-based reverse diffusion sampling for large 3D volumes.

Memory-safe alternative to running DiffusionSchedule.sample() on the full
volume. Reuses MONAI's sliding-window machinery but with a custom predictor
that keeps the noisy mask y_t spatially consistent across patches.
"""


import torch


def _gaussian_importance_map(
    patch_size: tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device | None = None,
) -> torch.Tensor:
    """3D Gaussian importance map, normalised to max=1, shape (1,1,*patch)."""

    coords = [torch.arange(s, dtype=torch.float32) for s in patch_size]
    grids = torch.meshgrid(*coords, indexing="ij")
    center = [(s - 1) / 2.0 for s in patch_size]
    sigmas = [s * sigma_scale for s in patch_size]
    g = torch.ones_like(grids[0])
    for gi, c, s in zip(grids, center, sigmas):
        g = g * torch.exp(-((gi - c) ** 2) / (2.0 * s ** 2))
    g = g / g.max()
    g = g.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    if device is not None:
        g = g.to(device)
    return g


def _extract_patches(vol: torch.Tensor, roi: tuple[int, int, int], overlap: float):
    """Yield (slice_tuple, patch) covering vol with given overlap. vol is (D,H,W)."""
    D, H, W = vol.shape
    stride = [max(1, round(r * (1.0 - overlap))) for r in roi]
    starts = []
    for size, s, r in zip((D, H, W), stride, roi):
        if size <= r:
            starts.append([0])
        else:
            ids = list(range(0, size - r + 1, s))
            if ids[-1] != size - r:
                ids.append(size - r)
            starts.append(ids)

    for dz in starts[0]:
        for dy in starts[1]:
            for dx in starts[2]:
                sl = (slice(dz, dz + roi[0]),
                      slice(dy, dy + roi[1]),
                      slice(dx, dx + roi[2]))
                yield sl


@torch.no_grad()
def patch_diffusion_sample(
    schedule,
    refiner,
    image: torch.Tensor,                 # (B,1,D,H,W)
    initial_logits: torch.Tensor,        # (B,1,D,H,W)
    roi_size: tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.25,
    inference_steps: int | None = None,
    blend: str = "gaussian",             # "gaussian" | "constant"
    seed: int | None = 1234,
) -> torch.Tensor:
    """Patch-wise DDPM reverse sampling.

    Returns a tensor of the same shape as `initial_logits`, usable directly
    with the same threshold as the baseline.
    """
    assert image.shape == initial_logits.shape
    # B, _C, _D, _H, _W = image.shape
    B, *_ = image.shape
    device = image.device

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Current noisy mask y_t initialised once over the whole volume
    current = torch.randn_like(initial_logits)

    # Choose weights for overlap blending
    if blend == "gaussian":
        wmap = _gaussian_importance_map(roi_size, device=device)  # (1,1,*roi)
    else:
        wmap = torch.ones((1, 1, *roi_size), device=device)

    # Reverse timesteps
    steps = list(reversed(range(schedule.timesteps)))
    if inference_steps is not None and inference_steps < schedule.timesteps:
        stride = max(1, schedule.timesteps // inference_steps)
        steps = list(reversed(range(0, schedule.timesteps, stride)))

    for step in steps:
        accum = torch.zeros_like(current)
        weight = torch.zeros_like(current)

        for sl in _extract_patches(current[0, 0], roi_size, overlap):
            sl_full = (slice(None), slice(None), *sl)  # add batch/channel
            img_p = image[sl_full]
            logits_p = initial_logits[sl_full]
            yt_p = current[sl_full]

            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps_hat = refiner(img_p, logits_p, yt_p, t)

            beta = schedule.betas[step]
            alpha = schedule.alphas[step]
            a_bar = schedule.alpha_bars[step]
            mean = (yt_p - ((1.0 - alpha) / (1.0 - a_bar).sqrt()) * eps_hat) / alpha.sqrt()

            if step > 0:
                yt_next = mean + beta.sqrt() * torch.randn_like(mean)
            else:
                yt_next = mean

            accum[sl_full] += yt_next * wmap
            weight[sl_full] += wmap

        current = accum / weight.clamp_min(1e-6)

    return current