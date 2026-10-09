from lungseg.models import ResUNet3D, build_segmentation_model


def test_resunet_output_shape(sample_batch):
    model = ResUNet3D(channels=(8, 16, 32, 64))
    out = model(sample_batch["image"])
    assert out.shape == sample_batch["label"].shape


def test_reunet_residual_skip_gradients(sample_batch):
    model = ResUNet3D(channels=(8, 16, 32, 64))
    out = model(sample_batch["image"])
    loss = out.mean()
    loss.backward()
    stem_grad = model.stem.weight.grad
    assert stem_grad is not None
    assert stem_grad.abs().sum().item() > 0


def test_factory_returns_correct_type():
    m = build_segmentation_model("resunet3d", channels=(8, 16, 32, 64))
    assert isinstance(m, ResUNet3D)