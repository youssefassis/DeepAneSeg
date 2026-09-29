"""Regression tests for bugs that silently changed training or sampling behaviour."""

import numpy as np
import pytest
import torch

from deepaneseg.training.dataset import get_number_of_steps
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


def make_trainer(config, model=None, optimizer=None, scheduler=None, loaders=None):
    model = model or torch.nn.Linear(2, 1)
    return create_trainer(
        **config,
        device="cpu",
        model=model,
        optimizer=optimizer or torch.optim.Adam(model.parameters()),
        lr_scheduler=scheduler,
        loss_criterion=None,
        eval_criterion=None,
        loaders=loaders or {"train": [], "valid": []},
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

    trainer._log_stats("train", loss_avg=0.5, metric_avg=0.7, step=0)

    assert any((tmp_path / "logs").rglob("events.out.tfevents.*"))


def test_resuming_restores_scheduler_and_early_stop_state(trainer_config):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    trainer = make_trainer(trainer_config, model, optimizer, scheduler)
    trainer.adjust_lr = True
    for score, loss in [(0.5, 1.0), (0.4, 2.0), (0.4, 2.0)]:  # best, then two epochs without improvement
        trainer._update_lr(loss)  # same order as Trainer.validate
        trainer._save_best("valid", score)

    new_optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(new_optimizer, factor=0.5, patience=2)
    resumed = make_trainer(trainer_config, model, new_optimizer, new_scheduler)

    assert resumed.nonimproved_epoch == 2
    assert new_scheduler.num_bad_epochs == scheduler.num_bad_epochs == 2
    assert new_scheduler.best == scheduler.best


def test_training_without_validation_set_still_saves_checkpoints(trainer_config, tmp_path):
    trainer = make_trainer(trainer_config, loaders={"train": [], "valid": None})

    assert trainer._save_best("train", 0.5)
    assert (tmp_path / "best_checkpoint.pytorch").exists()


@pytest.mark.parametrize("n_samples, batch_size, steps", [(5, 20, 1), (40, 20, 2), (41, 20, 3), (0, 20, 0)])
def test_number_of_steps_is_the_number_of_batches(n_samples, batch_size, steps):
    assert get_number_of_steps(n_samples, batch_size) == steps
