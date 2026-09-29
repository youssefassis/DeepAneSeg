import pytest
import torch

from deepaneseg.models.models import Proposition3, UNet3D


@pytest.fixture
def patch():
    return torch.randn(2, 1, 48, 48, 48)


def test_unet3d_outputs_probabilities_with_input_shape(patch):
    out = UNet3D(f_maps=8)(patch)

    assert out.shape == patch.shape
    assert out.min() >= 0 and out.max() <= 1


def test_proposition3_deep_supervision_returns_one_output_per_level(patch):
    outputs = Proposition3(f_maps=8)(patch)

    assert [o.shape[-1] for o in outputs] == [6, 12, 24, 48]
