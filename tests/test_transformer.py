import torch

from lungseg.config import load_config
from lungseg.models import (
    SwinUNETRSeg,
    UNETRSeg,
    build_from_config,
    build_segmentation_model,
)


def test_swinunetr_output_shape():
    # Spatial dims must be multiples of 32 AND >= 64
    # (at 32 the bottleneck is 1x1x1 and InstanceNorm fails in train mode).
    model = SwinUNETRSeg(feature_size=12).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64, 64))
    assert out.shape == (1, 1, 64, 64, 64)


def test_swinunetr_training_step(train_step_check):    
    train_step_check(SwinUNETRSeg(feature_size=12), 64)


def test_unetr_output_shape():
    model = build_segmentation_model("unetr", patch_size=(32, 32, 32)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 32, 32, 32))
    assert out.shape == (1, 1, 32, 32, 32)


def test_unetr_training_step(train_step_check):
    train_step_check(UNETRSeg(img_size=(32, 32, 32)), 32)


def test_configs_build_the_right_variant():
    assert isinstance(
        build_from_config(load_config("configs/experiments/e13_swinunetr.yaml")),
        SwinUNETRSeg,
    )
    cfg = load_config("configs/experiments/e13_unetr.yaml")
    cfg["data"]["patch_size"] = [32, 32, 32]     
    assert isinstance(build_from_config(cfg), UNETRSeg)