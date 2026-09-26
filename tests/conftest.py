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