import torch
from lungseg.diffusion import DiffusionSchedule


def test_add_noise_preserves_shape(sample_mask):
    sched = DiffusionSchedule(timesteps=10)
    noise = torch.randn_like(sample_mask)
    t = torch.zeros(sample_mask.shape[0], dtype=torch.long)
    noisy = sched.add_noise(sample_mask, noise, t)
    assert noisy.shape == sample_mask.shape


def test_add_noise_changes_mask(sample_mask):
    sched = DiffusionSchedule(timesteps=10)
    noise = torch.randn_like(sample_mask)
    t = torch.full((sample_mask.shape[0],), 9, dtype=torch.long)
    noisy = sched.add_noise(sample_mask, noise, t)
    assert not torch.allclose(noisy, sample_mask)