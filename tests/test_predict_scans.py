import json
import subprocess
import sys
from pathlib import Path

import nibabel as ni
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deepaneseg.inference.prediction import find_detections, save_detections
from deepaneseg.utils import get_model

ROOT = Path(__file__).parents[1]
AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])
AFFINE[:3, 3] = [-10, 20, 5]


def test_find_detections_reports_each_component_in_mm():
    prediction = np.zeros((40, 40, 40))
    prediction[10:14, 10:14, 10:14] = 0.9  # 64 voxels, 2 mm wide
    prediction[30, 30, 30] = 0.7  # single voxel
    prediction[20, 5, 5] = 0.3  # below threshold

    detections = find_detections(prediction, AFFINE, threshold=0.5)

    assert len(detections) == 2
    big = detections.sort_values("voxels").iloc[-1]
    np.testing.assert_allclose(big[["x", "y", "z"]].to_numpy(float), [-10 + 5.75, 20 + 5.75, 5 + 5.75])
    assert (big["voxels"], big["radius"], big["probability"]) == (64, pytest.approx(0.75), pytest.approx(0.9))
    assert len(find_detections(prediction, AFFINE, threshold=0.5, min_size=2)) == 1


def test_save_detections_writes_csv_and_slicer_markups(tmp_path):
    detections = find_detections(np.pad(np.ones((2, 2, 2)), 3), AFFINE)

    save_detections(detections, str(tmp_path / "scan_detections.csv"))

    assert len(pd.read_csv(tmp_path / "scan_detections.csv")) == 1
    markups = (tmp_path / "scan_detections.fcsv").read_text().splitlines()
    assert markups[0].startswith("# Markups fiducial file") and len(markups) == 4  # 3 header lines + 1 point


@pytest.fixture
def trained(tmp_path):
    """A tiny untrained model with its saved training configuration."""
    with initialize_config_dir(config_dir=str(ROOT / "deepaneseg" / "configs"), version_base="1.3"):
        cfg = compose(
            "train",
            [
                f"data_dir={tmp_path}",
                "name=exp",
                "model.f_maps=4",
                "data.patch_shape=[16,16,16]",
                "data.patch_size=[16,16,16]",
            ],
        )
    train_dir = tmp_path / "0Work" / "exp"
    (train_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, train_dir / ".hydra" / "config.yaml")
    torch.save({"model_state_dict": get_model(cfg.model).state_dict()}, train_dir / "last_checkpoint.pytorch")
    return train_dir


def predict(*args):
    return subprocess.run(
        [sys.executable, "-m", "deepaneseg.cli.predict", *map(str, args)], capture_output=True, text=True
    )


def write_scan(path, shape=(20, 18, 22)):
    ni.save(ni.Nifti1Image(np.random.default_rng(0).random(shape).astype(np.float32), AFFINE), path)


def test_predict_a_single_scan(trained, tmp_path):
    write_scan(tmp_path / "scan.nii.gz")

    result = predict(f"train_dir={trained}", f"input={tmp_path / 'scan.nii.gz'}", "skull_strip=false", "margin=2")

    assert result.returncode == 0, result.stderr[-2000:]
    output = trained / "prediction" / "scans"
    prediction = ni.load(output / "scan.nii.gz")
    assert prediction.shape == (20, 18, 22) and np.allclose(prediction.affine, AFFINE)
    assert (output / "scan_detections.csv").exists() and (output / "scan_detections.fcsv").exists()


def test_predict_a_folder_of_scans_into_a_chosen_folder(trained, tmp_path):
    (tmp_path / "scans").mkdir()
    write_scan(tmp_path / "scans" / "a.nii.gz")
    write_scan(tmp_path / "scans" / "b.nii")

    result = predict(
        f"train_dir={trained}",
        f"input={tmp_path / 'scans'}",
        f"output_dir={tmp_path / 'out'}",
        "skull_strip=false",
        "margin=2",
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert sorted(p.name for p in (tmp_path / "out").glob("*.nii.gz")) == ["a.nii.gz", "b.nii.gz"]


def test_predict_the_test_split_also_writes_detections(trained, tmp_path):
    patient = tmp_path / "P0001"
    patient.mkdir()
    write_scan(patient / "noskull.nii.gz")
    (patient / "config.json").write_text(
        json.dumps({"init volume": "noskull.nii.gz", "noskull volume": "noskull.nii.gz"})
    )
    (tmp_path / "0Work" / "split_pats.json").write_text(
        json.dumps({"training list": [], "validation list": [], "testing list": [str(patient)]})
    )

    result = predict(f"train_dir={trained}", "margin=2")

    assert result.returncode == 0, result.stderr[-2000:]
    output = trained / "prediction" / "test"
    assert (output / "P0001.nii.gz").exists() and (output / "P0001_detections.csv").exists()
