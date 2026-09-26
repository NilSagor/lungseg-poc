import torch

from lungseg.models import BaselineUNet, DiffusionRefiner


def test_baseline_output_shape(sample_batch):
    model = BaselineUNet()
    out = model(sample_batch["image"])
    assert out.shape == sample_batch["label"].shape


def test_refiner_output_shape(sample_batch):
    refiner = DiffusionRefiner()
    image = sample_batch["image"]
    logits = torch.zeros_like(sample_batch["label"])
    noisy = torch.randn_like(sample_batch["label"])
    t = torch.zeros(image.shape[0], dtype=torch.long)
    out = refiner(image, logits, noisy, t)
    assert out.shape == sample_batch["label"].shape