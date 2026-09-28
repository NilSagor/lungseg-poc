import torch

from lungseg.diffusion import DiffusionSchedule
from lungseg.inference.patch_diffusion import (
    _extract_patches,
    _gaussian_importance_map,
    patch_diffusion_sample,
)
from lungseg.models import DiffusionRefiner


def test_gaussian_map_shape_and_range():
    g = _gaussian_importance_map((16, 16, 16))
    assert g.shape == (1, 1, 16, 16, 16)
    assert g.max().item() <= 1.0 + 1e-6
    assert g.min().item() > 0.0


def test_extract_patches_covers_volume():
    vol = torch.zeros(40, 40, 40)
    seen = torch.zeros_like(vol)
    for sl in _extract_patches(vol, roi=(16, 16, 16), overlap=0.5):
        seen[sl] += 1
    assert (seen > 0).all(), "Every voxel must be visited at least once"


def test_patch_diffusion_sample_shape(sample_batch):
    schedule = DiffusionSchedule(timesteps=4).to(sample_batch["image"].device)
    refiner = DiffusionRefiner(
        channels=(8, 16, 32),
        num_res_units=1,
    ).to(sample_batch["image"].device)

    image = sample_batch["image"]
    logits = torch.zeros_like(sample_batch["label"])
    out = patch_diffusion_sample(
        schedule=schedule,
        refiner=refiner,
        image=image,
        initial_logits=logits,
        roi_size=(16, 16, 16),
        overlap=0.5,
        inference_steps=2,
        seed=0,
    )
    assert out.shape == logits.shape


def test_patch_diffusion_is_deterministic_given_seed(sample_batch):
    schedule = DiffusionSchedule(timesteps=4).to(sample_batch["image"].device)
    refiner = DiffusionRefiner(channels=(8, 16, 32), num_res_units=1).to(
        sample_batch["image"].device
    )
    refiner.eval()
    image = sample_batch["image"]
    logits = torch.zeros_like(sample_batch["label"])

    a = patch_diffusion_sample(schedule, refiner, image, logits,
                               roi_size=(16, 16, 16), overlap=0.5,
                               inference_steps=2, seed=7)
    b = patch_diffusion_sample(schedule, refiner, image, logits,
                               roi_size=(16, 16, 16), overlap=0.5,
                               inference_steps=2, seed=7)
    assert torch.allclose(a, b)