#!/usr/bin/env python3
"""Train a model; configuration in configs/train.yaml.

Example: python scripts/train.py data_dir=/data name=exp1 batch_size=8
Checkpoints, TensorBoard logs and the resolved configuration (.hydra/) are written to <data_dir>/0Work/<name>.
Running the same command again resumes from the last checkpoint.
"""

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from deepaneseg.data.io import add_points_to_patient_data, read_patient_data_base, read_split
from deepaneseg.training.dataset import get_dataloaders
from deepaneseg.training.losses import get_loss_criterion
from deepaneseg.training.metrics import get_metric
from deepaneseg.training.trainer import create_trainer
from deepaneseg.utils import get_model


def load_patients(pat_list, data_cfg, label):
    if not pat_list:
        return None
    patients = read_patient_data_base(pat_list, volume=data_cfg.volume, normalize=data_cfg.normalize, label=label)
    return add_points_to_patient_data(patients, data_cfg.negative_patch_centers)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg):
    train_dir = HydraConfig.get().runtime.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg.model, weight_init=cfg.weight_init, device=device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters(), _convert_="all")
    lr_scheduler = instantiate(cfg.scheduler, optimizer=optimizer, _convert_="all")

    train_list, valid_list, _ = read_split(cfg.data.split_file)
    train_db = load_patients(train_list, cfg.data, "training")
    valid_db = load_patients(valid_list, cfg.data, "validation")
    loaders, iterations = get_dataloaders(train_db, valid_db, cfg)

    trainer = create_trainer(
        train_dir,
        max_num_epochs=cfg.epochs,
        early_stop=cfg.early_stop,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=get_loss_criterion(cfg.loss),
        eval_criterion=get_metric(cfg.metric),
        loaders=loaders,
        max_iterations=iterations,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
