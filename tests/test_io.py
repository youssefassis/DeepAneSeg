import json

import numpy as np
import pytest

from deepaneseg.data import io as dio


def test_read_points_from_csv_returns_none_for_missing_file(tmp_path):
    assert dio.read_points_from_csv(str(tmp_path / "missing.csv")) is None


def test_read_points_from_csv_rejects_file_without_coordinates(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")

    with pytest.raises(ValueError):
        dio.read_points_from_csv(str(bad))


def test_read_points_from_csv_reads_xyz(tmp_path):
    f = tmp_path / "F.csv"
    f.write_text("label,x,y,z\na,1,2,3\nb,4,5,6\n")

    np.testing.assert_array_equal(dio.read_points_from_csv(str(f)), [[1, 2, 3], [4, 5, 6]])


def test_save_split_creates_file_and_keeps_existing_keys(tmp_path):
    split = tmp_path / "split.json"
    dio.save_split(str(split), ["P0001"], ["P0002"], ["P0003"])
    split.write_text(json.dumps({**json.loads(split.read_text()), "note": "kept"}))

    dio.save_split(str(split), ["P0004"], [], [])

    content = json.loads(split.read_text())
    assert content["training list"] == ["P0004"] and content["note"] == "kept"
    assert dio.read_split(str(split)) == (["P0004"], [], [])
