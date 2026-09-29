#!/usr/bin/env python3
"""Predict the test patients with a trained model; configuration in configs/predict.yaml.

Example: python scripts/predict.py train_dir=/data/0Work/exp1
Data and model settings come from the training's resolved configuration (<train_dir>/.hydra/config.yaml).
Predicted volumes are written to <train_dir>/prediction/test.
"""

import os

import hydra
import torch
from omegaconf import OmegaConf

import deepaneseg.data.io as dio
from deepaneseg.inference.prediction import ndl_run_validation_cases
from deepaneseg.utils import get_logger, load_model


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict")
def main(cfg):
    logger = get_logger("Prediction")
    train_cfg = OmegaConf.load(os.path.join(cfg.train_dir, ".hydra", "config.yaml"))

    _, _, test_list = dio.read_split(train_cfg.data.split_file)
    logger.info(f"{len(test_list)} Patients for testing")
    test_db = dio.read_patient_data_base(test_list, volume=cfg.volume, normalize=train_cfg.data.normalize)
    logger.info(f"'{cfg.volume}' successfully loaded with '{train_cfg.data.normalize}' normalization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(train_cfg.model, os.path.join(cfg.train_dir, cfg.checkpoint)).to(device)
    model.eval()
    logger.info(f"Model '{cfg.checkpoint}' loaded on '{device}' in evaluation mode")

    ndl_run_validation_cases(
        test_db,
        device=device,
        model=model,
        patch_size=list(train_cfg.data.patch_size),
        output_dir=os.path.join(cfg.train_dir, "prediction", "test"),
        margin=cfg.margin,
        batch_size=cfg.batch_size or train_cfg.validation_batch_size,
    )


if __name__ == "__main__":
    main()
