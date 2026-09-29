#!/usr/bin/env python3
"""Score a training's predictions on its test patients; configuration in configs/evaluate.yaml.

Example: python scripts/evaluate.py train_dir=/data/0Work/exp1
A detection (connected component above the threshold) is a true positive when its center lies within an
aneurysm's radius (ADAM challenge criteria). Writes per_patient.csv and summary.json to <train_dir>/evaluation.
"""

import json
import os

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import deepaneseg.data.io as dio
from deepaneseg.inference.evaluation import adam_evaluation, summarize_detections
from deepaneseg.utils import get_logger


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate")
def main(cfg):
    logger = get_logger("Evaluation")
    output_dir = HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.load(os.path.join(cfg.train_dir, ".hydra", "config.yaml"))
    _, _, test_list = dio.read_split(train_cfg.data.split_file)

    files = {os.path.basename(p): os.path.join(cfg.predictions, os.path.basename(p) + ".nii.gz") for p in test_list}
    missing = sorted(name for name, f in files.items() if not os.path.isfile(f))
    if missing:
        raise FileNotFoundError(f"No prediction in {cfg.predictions} for {', '.join(missing)}; run predict.py first")

    rows = {}
    for patient_dir in test_list:
        name = os.path.basename(patient_dir)
        prediction, affine = dio.read_nii_from_file(files[name])
        spheres = dio.read_aneurysm_spheres(patient_dir)
        tp, fp, _, _ = adam_evaluation(
            np.where(prediction >= cfg.threshold, 1, 0), affine, spheres, min_size=cfg.min_size
        )
        rows[name] = {"aneurysms": len(spheres), "tp": tp, "fn": len(spheres) - tp, "fp": fp}

    per_patient = pd.DataFrame.from_dict(rows, orient="index").rename_axis("patient")
    summary = summarize_detections(per_patient)
    per_patient.to_csv(os.path.join(output_dir, "per_patient.csv"))
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Detection metrics: {summary}")


if __name__ == "__main__":
    main()
