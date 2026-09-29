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

    first = run("preprocess_data.py", tmp_path)
    second = run("preprocess_data.py", tmp_path)

    assert first.returncode == 0, first.stderr
    split = json.loads((tmp_path / "0Work" / "BaseConfig" / "split_pats.json").read_text())
    assert [len(split[k]) for k in ("training list", "validation list", "testing list")] == [7, 2, 1]
    assert second.returncode != 0 and "--overwrite" in second.stderr


def test_new_train_creates_training_directory(tmp_path):
    make_patients(tmp_path, 3)
    run("preprocess_data.py", tmp_path)

    result = run("new_train.py", tmp_path / "0Work" / "BaseConfig" / "ndl_config.json", "exp1")

    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "0Work" / "exp1" / "ndl_config.json").read_text())
    assert config["model"] == "unet3d" and config["test_dir"] == str(tmp_path.resolve() / "0Work" / "exp1")


def test_train_reports_missing_config(tmp_path):
    result = run("train.py", tmp_path)

    assert result.returncode != 0 and "ndl_config.json" in result.stderr
