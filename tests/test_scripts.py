import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True)


def make_patients(data_dir, n):
    for i in range(1, n + 1):
        (data_dir / f"P{i:04d}").mkdir()
        (data_dir / f"P{i:04d}" / "config.json").write_text("{}")


def test_preprocess_data_splits_patients_and_refuses_to_overwrite(tmp_path):
    make_patients(tmp_path, 10)

    first = run("preprocess_data.py", f"data_dir={tmp_path}")
    second = run("preprocess_data.py", f"data_dir={tmp_path}")

    assert first.returncode == 0, first.stderr
    split = json.loads((tmp_path / "0Work" / "split_pats.json").read_text())
    assert [len(split[k]) for k in ("training list", "validation list", "testing list")] == [7, 2, 1]
    assert second.returncode != 0 and "overwrite=true" in second.stderr


def test_train_requires_data_dir_and_name():
    result = run("train.py")

    assert result.returncode != 0 and "Missing mandatory value" in result.stderr


def test_predict_uses_the_volume_the_model_was_trained_on(tmp_path):
    import nibabel as ni
    import numpy as np
    import pandas as pd
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from deepaneseg.utils import get_model

    patient = tmp_path / "P0001"  # raw volume only: no skull stripping was run
    patient.mkdir()
    ni.save(
        ni.Nifti1Image(np.random.default_rng(0).random((20, 20, 20)).astype(np.float32), np.eye(4)),
        patient / "volume.nii.gz",
    )
    pd.DataFrame({"x": [9.0, 11.0], "y": [10.0, 10.0], "z": [10.0, 10.0]}).to_csv(patient / "F.csv", index=False)
    (patient / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz", "pts aneurysm": "F.csv"}))

    with initialize_config_dir(config_dir=str(SCRIPTS.parent / "configs"), version_base="1.3"):
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
    (tmp_path / "0Work" / "split_pats.json").write_text(
        json.dumps({"training list": [], "validation list": [], "testing list": [str(patient)]})
    )
    torch.save({"model_state_dict": get_model(cfg.model).state_dict()}, train_dir / "last_checkpoint.pytorch")

    result = run("predict.py", f"train_dir={train_dir}", "margin=2", "batch_size=4")

    assert result.returncode == 0, result.stderr[-2000:]
    assert ni.load(train_dir / "prediction" / "test" / "P0001.nii.gz").shape == (20, 20, 20)
