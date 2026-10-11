import pytest
import torch


@pytest.fixture
def sample_batch():
    image = torch.randn(2, 1, 32, 32, 32)
    label = (torch.rand(2, 1, 32, 32, 32) > 0.95).float()
    return {"image": image, "label": label}


@pytest.fixture
def sample_mask():
    return (torch.rand(2, 1, 32, 32, 32) > 0.9).float()


@pytest.fixture
def train_step_check():
    """Run one real optimisation step (loss incl. deep supervision) and assert
    the loss is finite and the weights moved."""
    from lungseg.losses import BCEDiceLoss, deep_supervision_loss

    def _check(model, size):
        torch.manual_seed(0)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(2, 1, size, size, size)
        y = (torch.rand(2, 1, size, size, size) > 0.9).float()
        before = [p.detach().clone() for p in model.parameters()]
        loss = deep_supervision_loss(BCEDiceLoss(), model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))

    return _check