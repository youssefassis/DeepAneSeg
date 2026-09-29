import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from deepaneseg.training.losses import FocalLoss, KappaLoss
from deepaneseg.training.metrics import Kappa


@pytest.fixture
def binary_pair():
    g = torch.Generator().manual_seed(0)
    target = (torch.rand(2, 1, 16, 16, 16, generator=g) < 0.1).float()
    flip = torch.rand(target.shape, generator=g) < 0.2
    return torch.where(flip, 1 - target, target), target


def test_kappa_matches_cohens_kappa(binary_pair):
    pred, target = binary_pair

    expected = cohen_kappa_score(target.flatten().numpy(), pred.flatten().numpy())

    assert Kappa()(pred, target).item() == pytest.approx(expected, abs=1e-6)
    assert 1 - KappaLoss()(pred, target).item() == pytest.approx(expected, abs=1e-6)


def test_focal_loss_weights_each_voxel():
    probs = torch.tensor([0.9, 0.1, 0.6, 0.3])
    target = torch.tensor([1.0, 0.0, 0.0, 1.0])
    alpha, gamma = 0.8, 2

    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = torch.where(target == 1, probs, 1 - probs)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    expected = (alpha_t * (1 - p_t) ** gamma * bce).mean()

    assert FocalLoss()(probs, target).item() == pytest.approx(expected.item(), rel=1e-6)


def test_focal_loss_downweights_easy_voxels_more_than_bce():
    easy = torch.tensor([0.99, 0.01])
    target = torch.tensor([1.0, 0.0])

    assert FocalLoss()(easy, target) < 1e-3 * F.binary_cross_entropy(easy, target)
