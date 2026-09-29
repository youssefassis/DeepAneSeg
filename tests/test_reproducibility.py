import json
import random
import subprocess
import sys
from pathlib import Path

import nibabel as ni
import numpy as np
import pandas as pd
import pytest
import torch

from deepaneseg.data.generators import split_pat_list
from deepaneseg.data.io import fetch_patient_dirs
from deepaneseg.utils import run_info, seed_everything

SCRIPTS = Path(__file__).parents[1] / "scripts"


def test_split_depends_only_on_the_seed():
    patients = [f"P{i:04d}" for i in range(20)]
    shuffled = random.sample(patients, len(patients))

    first = split_pat_list(patients, 0.7, 0.2, 0.1, seed=3)

    assert split_pat_list(shuffled, 0.7, 0.2, 0.1, seed=3) == first
    assert split_pat_list(patients, 0.7, 0.2, 0.1, seed=4) != first
    assert patients == sorted(patients)  # the input list is left untouched


def test_fetch_patient_dirs_is_sorted(tmp_path):
    for name in ("P0003", "P0001", "P0002"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")

    assert [Path(p).name for p in fetch_patient_dirs(str(tmp_path))] == ["P0001", "P0002", "P0003"]


def test_seed_everything_seeds_python_numpy_and_torch():
    def draws():
        return random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(7)
    first = draws()
    seed_everything(7)

    assert draws() == first


def test_run_info_records_versions_and_commit():
    info = run_info()

    assert info["torch"] == torch.__version__
    assert info["git_commit"] != "unknown"  # the tests run from the git checkout


def make_dataset(data_dir):
    rng = np.random.default_rng(0)
    patients = []
    for i in range(1, 4):
        patient = data_dir / f"P{i:04d}"
        patient.mkdir()
        ni.save(ni.Nifti1Image(rng.random((24, 24, 24)).astype(np.float32), np.eye(4)), patient / "volume.nii.gz")
        pd.DataFrame({"x": [10.0, 12.0], "y": [12.0, 12.0], "z": [12.0, 12.0]}).to_csv(patient / "F.csv", index=False)
        pd.DataFrame({"x": [6.0, 18.0], "y": [6.0, 18.0], "z": [6.0, 18.0]}).to_csv(patient / "points.csv", index=False)
        (patient / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz", "pts aneurysm": "F.csv"}))
        patients.append(str(patient))
    (data_dir / "0Work").mkdir()
    (data_dir / "0Work" / "split_pats.json").write_text(
        json.dumps({"training list": patients[:2], "validation list": patients[2:], "testing list": []})
    )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_same_seed_gives_the_same_trained_weights(tmp_path, num_workers):
    make_dataset(tmp_path)

    def train(name):
        args = [
            f"data_dir={tmp_path}",
            f"name={name}",
            "seed=5",
            "epochs=1",
            "batch_size=2",
            "validation_batch_size=2",
            f"num_workers={num_workers}",
            "model.f_maps=4",
            "data.patch_shape=[16,16,16]",
            "data.patch_size=[16,16,16]",
            "augmentation.positive.duplicates=2",
        ]
        result = subprocess.run([sys.executable, str(SCRIPTS / "train.py"), *args], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-2000:]
        return torch.load(tmp_path / "0Work" / name / "last_checkpoint.pytorch")["model_state_dict"]

    first, second = train("a"), train("b")

    assert all(torch.equal(first[k], second[k]) for k in first)
    assert json.loads((tmp_path / "0Work" / "a" / "run_info.json").read_text())["seed"] == 5
