"""Predict with a trained model; configuration in deepaneseg/configs/predict.yaml.

Examples:
    deepaneseg-predict train_dir=/data/0Work/exp1                      # test patients of the split
    deepaneseg-predict train_dir=/data/0Work/exp1 input=scan.nii.gz    # any scan, or a folder of scans
Data and model settings come from the training's resolved configuration (<train_dir>/.hydra/config.yaml).
For each case, writes the probability map <name>.nii.gz and the detections <name>_detections.csv (and .fcsv,
for 3D Slicer).
"""

import glob
import os

import hydra
import torch
from omegaconf import OmegaConf

import deepaneseg.data.io as dio
from deepaneseg.inference.prediction import find_detections, predict_volume, save_detections
from deepaneseg.utils import get_logger, load_model
from deepaneseg.volume.selection import skull_strip

logger = get_logger("Prediction")


def scan_name(path):
    return os.path.basename(path).removesuffix(".gz").removesuffix(".nii")


def input_scans(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.nii")) + glob.glob(os.path.join(path, "*.nii.gz")))
        if not files:
            raise FileNotFoundError(f"No .nii or .nii.gz file in {path}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such scan or folder: {path}")
    return [path]


def cases(cfg, normalize):
    """Yields (name, normalized volume, affine) for the input scans or the test patients."""
    if cfg.input is not None:
        for path in input_scans(cfg.input):
            data, affine = dio.read_nii_from_file(path)
            if cfg.skull_strip:
                data = skull_strip(data)
            yield scan_name(path), dio.normalize_volume(data, normalize), affine
        return
    train_cfg = OmegaConf.load(os.path.join(cfg.train_dir, ".hydra", "config.yaml"))
    _, _, test_list = dio.read_split(train_cfg.data.split_file)
    logger.info(f"{len(test_list)} patients for testing, '{cfg.volume}' with '{normalize}' normalization")
    for patient in dio.read_patient_data_base(test_list, volume=cfg.volume, normalize=normalize):
        yield os.path.basename(patient["dir"]), patient["data"], patient["affine"]


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict")
def main(cfg):
    train_cfg = OmegaConf.load(os.path.join(cfg.train_dir, ".hydra", "config.yaml"))
    default_output = os.path.join(cfg.train_dir, "prediction", "test" if cfg.input is None else "scans")
    output_dir = cfg.output_dir or default_output
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(train_cfg.model, os.path.join(cfg.train_dir, cfg.checkpoint)).to(device)
    model.eval()
    logger.info(f"Model '{cfg.checkpoint}' loaded on '{device}' in evaluation mode")

    for name, data, affine in cases(cfg, train_cfg.data.normalize):
        prediction = predict_volume(
            data,
            affine,
            device,
            model,
            patch_size=list(train_cfg.data.patch_size),
            patch_shape=list(train_cfg.data.patch_shape),
            margin=cfg.margin,
            batch_size=cfg.batch_size or train_cfg.validation_batch_size,
            name=name,
        )
        dio.save_nii_to_file(os.path.join(output_dir, name + ".nii.gz"), prediction, affine)
        detections = find_detections(prediction, affine, threshold=cfg.threshold, min_size=cfg.min_size)
        save_detections(detections, os.path.join(output_dir, name + "_detections.csv"))
        logger.info(f"{name}: {len(detections)} detections written to {output_dir}")


if __name__ == "__main__":
    main()
