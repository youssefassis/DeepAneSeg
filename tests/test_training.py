import pytest
import torch

from deepaneseg.training.losses import DiceLoss
from deepaneseg.training.metrics import DiceCoefficient
from deepaneseg.utils import RunningAverage


def test_dice_is_one_for_perfect_prediction():
    target = (torch.rand(1, 1, 8, 8, 8) > 0.7).float()

    assert DiceCoefficient()(target, target).item() == pytest.approx(1.0)
    assert DiceLoss()(target, target).item() == pytest.approx(0.0)


def test_running_average_weights_by_count():
    avg = RunningAverage()
    avg.update(1.0, n=1)
    avg.update(4.0, n=3)

    assert avg.avg == pytest.approx(13 / 4)
