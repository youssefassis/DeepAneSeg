#!/usr/bin/env python3
"""Predict the test patients with the model of <train_dir>; volumes are written to <train_dir>/prediction/test."""

import argparse
import json
import os

import deepaneseg.data.io as dio
import torch

from deepaneseg.utils import get_logger, load_model
from deepaneseg.inference.prediction import ndl_run_validation_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", help="training directory containing ndl_config.json (see new_train.py)")
    d = parser.parse_args().train_dir
    with open(os.path.join(d, "ndl_config.json")) as f:
        config = json.load(f)

    logger = get_logger("Data Preparation")

    normalize = config["normalize"] if "normalize" in config else None

    _, _, test_list = dio.read_split(config["split_file"])
    logger.info(f"{len(test_list)} Patients for testing")

    volume = "noskull volume"  # or init volume

    test_db = dio.read_patient_data_base(test_list, volume=volume, normalize=normalize)
    logger.info(f"'{volume}' successfully loaded with '{normalize}' normalization")

    logger = get_logger("Model")

    model = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending the model to '{device}'")

    model.eval()
    logger.info("Setting the model to evaluation mode")

    ndl_run_validation_cases(
        test_db,
        device=device,
        model=model,
        patch_size=config["patch_size"],
        output_dir=os.path.join(d, "prediction/test"),
        margin=8,
        batch_size=config["validation_batch_size"],
    )


if __name__ == "__main__":
    main()
