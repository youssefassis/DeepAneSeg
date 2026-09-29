"""The Hydra configs compose, and what they instantiate trains and resumes."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

from deepaneseg.models.models import UNet3D
from deepaneseg.training.trainer import create_trainer
from deepaneseg.utils import get_model

CONFIG_DIR = str(Path(__file__).parents[1] / "configs")


def compose_config(name, overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=name, overrides=overrides)


@pytest.fixture
def train_cfg(tmp_path):
    return compose_config("train", [f"data_dir={tmp_path}", "name=exp", "model.f_maps=4"])


def test_train_config_derives_paths_from_data_dir(train_cfg, tmp_path):
    assert train_cfg.data.work_dir == f"{tmp_path}/0Work"
    assert train_cfg.data.split_file == f"{tmp_path}/0Work/split_pats.json"


def test_train_config_defaults_match_the_paper_setup(train_cfg):
    assert list(train_cfg.data.patch_shape) == [48, 48, 48]
    assert list(train_cfg.data.patch_size) == [38, 38, 38]
    assert train_cfg.augmentation.positive.duplicates == 50
    assert (train_cfg.loss, train_cfg.metric) == ("BCELoss", "Kappa")


@pytest.mark.parametrize("name", ["train", "preprocess", "remove_skull", "extract_points"])
def test_configs_require_data_dir(name):
    cfg = compose_config(name, [])

    with pytest.raises(Exception, match="data_dir"):
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def test_predict_config_requires_train_dir():
    with pytest.raises(Exception, match="train_dir"):
        OmegaConf.to_container(compose_config("predict", []), resolve=True, throw_on_missing=True)


def test_instantiated_training_can_resume_from_its_checkpoint(train_cfg, tmp_path):
    model = get_model(train_cfg.model)
    optimizer = instantiate(train_cfg.optimizer, params=model.parameters(), _convert_="all")
    scheduler = instantiate(train_cfg.scheduler, optimizer=optimizer, _convert_="all")
    assert isinstance(model, UNet3D)
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert scheduler.patience == 10

    def trainer():
        return create_trainer(
            str(tmp_path), 1, train_cfg.early_stop, "cpu", model, optimizer, scheduler, None, None, None, None
        )

    trainer()._save_checkpoint(is_best=True)
    resumed = trainer()  # loads last_checkpoint.pytorch with torch.load's default weights_only=True

    assert resumed.num_epoch == 1
    betas = optimizer.param_groups[0]["betas"]
    assert not isinstance(betas, ListConfig) and tuple(betas) == (0.9, 0.999)
