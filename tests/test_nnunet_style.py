import pytest
import torch

from lungseg.config import load_config
from lungseg.losses import deep_supervision_loss
from lungseg.models import NNUNetStyle, build_from_config


def test_train_mode_returns_stacked_full_res_heads():
    model = NNUNetStyle(features=(8, 16, 32, 64)).train()
    out = model(torch.randn(2, 1, 32, 32, 32))
    # MONAI stacks heads on dim 1 and resamples every head to full resolution.
    assert out.shape == (2, 3, 1, 32, 32, 32)


def test_eval_mode_returns_plain_tensor(sample_batch):
    model = NNUNetStyle(features=(8, 16, 32, 64)).eval()
    with torch.no_grad():
        out = model(sample_batch["image"])
    assert out.shape == sample_batch["label"].shape


def test_deep_supervision_loss_weights_and_passthrough():
    def loss_fn(logits, target):
        return logits.mean()

    t = torch.zeros(1, 1, 4, 4, 4)

    # Plain tensor passes through unchanged.
    plain = torch.full((1, 1, 4, 4, 4), 2.0)
    assert deep_supervision_loss(loss_fn, plain, t).item() == 2.0

    # Stacked (B, N, C, D, H, W) with head means 4, 2, 1 -> weighted 1, .5, .25.
    stacked = torch.stack(
        [torch.full((1, 1, 4, 4, 4), v) for v in (4.0, 2.0, 1.0)], dim=1
    )
    expected = (4.0 + 0.5 * 2.0 + 0.25 * 1.0) / 1.75
    assert deep_supervision_loss(loss_fn, stacked, t).item() == pytest.approx(expected)

    # List of tensors takes the same path.
    assert deep_supervision_loss(
        loss_fn, list(stacked.unbind(1)), t
    ).item() == pytest.approx(expected)


def test_training_step_with_deep_supervision(train_step_check):
    # train_step_check comes from tests/conftest.py (Phase 11 fixture).
    train_step_check(NNUNetStyle(features=(8, 16, 32, 64)), 32)


def test_config_builds_nnunet_style():
    model = build_from_config(load_config("configs/experiments/e12_nnunet_style.yaml"))
    assert isinstance(model, NNUNetStyle)