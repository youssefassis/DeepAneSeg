"""Regression tests for bugs that silently changed training or sampling behaviour."""

import numpy as np
import pytest
import torch

from deepaneseg.training.trainer import create_trainer
from deepaneseg.volume.selection import points_in_radius, select_points

HALF_MM = np.diag([0.5, 0.5, 0.5, 1.0])


def test_select_points_stays_away_from_every_forbidden_point():
    vol = np.random.default_rng(0).random((40, 40, 40))
    forbidden = np.array([[5.0, 5, 5], [15.0, 15, 15]])

    points = select_points(vol, HALF_MM, thres_low=0.5, r=4, forbidden_points=forbidden)

    distances = np.linalg.norm(points[:, None] - forbidden[None], axis=-1)
    assert distances.min() > 4


def test_points_in_radius_accepts_several_queries():
    p = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])

    idx = points_in_radius(np.array([[0.0, 0, 0], [20.0, 0, 0]]), p, r=1)

    assert sorted(idx) == [0, 2]


@pytest.fixture
def trainer_config(tmp_path):
    return {"train_dir": str(tmp_path), "max_num_epochs": 1, "early_stop": 3}


def make_trainer(config):
    model = torch.nn.Linear(2, 1)
    return create_trainer(
        **config,
        device="cpu",
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        lr_scheduler=None,
        loss_criterion=None,
        eval_criterion=None,
        loaders=None,
        max_iterations=None,
    )


def test_trainer_uses_configured_early_stop(trainer_config):
    assert make_trainer(trainer_config).earlystop == 3


def test_early_stop_counter_resets_when_score_improves(trainer_config):
    trainer = make_trainer(trainer_config)

    for score in [0.5, 0.4, 0.3, 0.6]:  # two non-improving epochs, then a new best
        trainer._save_best("valid", score)

    assert trainer.nonimproved_epoch == 0


def test_training_stops_after_early_stop_epochs_without_improvement(trainer_config):
    trainer = make_trainer(trainer_config)

    for score in [0.5, 0.4, 0.4, 0.4]:
        trainer._save_best("valid", score)

    assert trainer.should_stop(train=False)


def test_log_stats_writes_tensorboard_scalars(trainer_config, tmp_path):
    trainer = make_trainer(trainer_config)

    trainer._log_stats("train", loss_avg=0.5, dice_avg=0.7, step=0)

    assert any((tmp_path / "logs").rglob("events.out.tfevents.*"))
