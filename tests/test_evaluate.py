import json
import subprocess
import sys
from pathlib import Path

import nibabel as ni
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from deepaneseg.data.io import read_aneurysm_spheres
from deepaneseg.inference.evaluation import summarize_detections

SCRIPTS = Path(__file__).parents[1] / "scripts"


def make_patient(data_dir, name, aneurysm_points=None):
    patient = data_dir / name
    patient.mkdir()
    config = {"init volume": "volume.nii.gz"}
    if aneurysm_points is not None:
        pd.DataFrame(aneurysm_points, columns=["x", "y", "z"]).to_csv(patient / "F.csv", index=False)
        config["pts aneurysm"] = "F.csv"
    (patient / "config.json").write_text(json.dumps(config))
    return patient


def test_read_aneurysm_spheres(tmp_path):
    ill = make_patient(tmp_path, "P0001", [[8.0, 10, 10], [12.0, 10, 10]])
    healthy = make_patient(tmp_path, "P0002")

    np.testing.assert_allclose(read_aneurysm_spheres(str(ill)), [[10, 10, 10, 2]])
    assert read_aneurysm_spheres(str(healthy)).shape == (0, 4)


def test_summarize_detections_averages_sensitivity_over_patients_with_aneurysms():
    per_patient = pd.DataFrame(
        {"aneurysms": [2, 1, 0], "tp": [1, 1, 0], "fn": [1, 0, 0], "fp": [0, 1, 2]},
        index=["P0001", "P0002", "P0003"],
    )

    summary = summarize_detections(per_patient)

    assert summary["sensitivity"] == pytest.approx((0.5 + 1.0) / 2)
    assert summary["sensitivity_pooled"] == pytest.approx(2 / 3)
    assert summary["fp_per_patient"] == pytest.approx(1.0)
    assert (summary["patients"], summary["aneurysms"]) == (3, 3)


def test_evaluate_script_scores_the_test_predictions(tmp_path):
    ill = make_patient(tmp_path, "P0001", [[8.0, 10, 10], [12.0, 10, 10]])  # aneurysm of radius 2 mm at (10, 10, 10)
    healthy = make_patient(tmp_path, "P0002")
    train_dir = tmp_path / "0Work" / "exp"
    predictions = train_dir / "prediction" / "test"
    predictions.mkdir(parents=True)
    (train_dir / ".hydra").mkdir()
    OmegaConf.save({"data": {"split_file": str(tmp_path / "split.json")}}, train_dir / ".hydra" / "config.yaml")
    (tmp_path / "split.json").write_text(
        json.dumps({"training list": [], "validation list": [], "testing list": [str(ill), str(healthy)]})
    )
    detected = np.zeros((20, 20, 20))
    detected[9:12, 9:12, 9:12] = 0.9  # on the aneurysm
    detected[2:4, 2:4, 2:4] = 0.8  # elsewhere
    ni.save(ni.Nifti1Image(detected, np.eye(4)), predictions / "P0001.nii.gz")
    ni.save(ni.Nifti1Image(np.full((20, 20, 20), 0.1), np.eye(4)), predictions / "P0002.nii.gz")

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "evaluate.py"), f"train_dir={train_dir}"], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr[-2000:]
    per_patient = pd.read_csv(train_dir / "evaluation" / "per_patient.csv", index_col="patient")
    assert per_patient.loc["P0001", ["aneurysms", "tp", "fn", "fp"]].tolist() == [1, 1, 0, 1]
    assert per_patient.loc["P0002", ["aneurysms", "tp", "fn", "fp"]].tolist() == [0, 0, 0, 0]
    summary = json.loads((train_dir / "evaluation" / "summary.json").read_text())
    assert (summary["sensitivity"], summary["fp_per_patient"]) == (1.0, 0.5)


def test_evaluate_script_reports_missing_predictions(tmp_path):
    ill = make_patient(tmp_path, "P0001", [[8.0, 10, 10], [12.0, 10, 10]])
    train_dir = tmp_path / "exp"
    (train_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save({"data": {"split_file": str(tmp_path / "split.json")}}, train_dir / ".hydra" / "config.yaml")
    (tmp_path / "split.json").write_text(
        json.dumps({"training list": [], "validation list": [], "testing list": [str(ill)]})
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "evaluate.py"), f"train_dir={train_dir}"], capture_output=True, text=True
    )

    assert result.returncode != 0 and "P0001" in result.stderr
