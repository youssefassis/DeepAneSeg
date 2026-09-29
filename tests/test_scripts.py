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
