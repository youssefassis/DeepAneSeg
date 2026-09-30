import subprocess
import sys

import nibabel as ni
import numpy as np
import pandas as pd
import pytest

from deepaneseg.data.io import fetch_patient_dirs, read_aneurysm_spheres, read_patient_data_base
from deepaneseg.data.prepare import aneurysm_points, find_cases
from deepaneseg.volume.selection import connected_components_to_spheres

AFFINE = np.diag([0.4, 0.4, 0.6, 1.0])
AFFINE[:3, 3] = [-50, -60, 10]


def write_adam_case(source, name, blobs=()):
    """ADAM challenge layout: <case>/orig/TOF.nii.gz and <case>/aneurysms.nii.gz (1: untreated, 2: treated)."""
    (source / name / "orig").mkdir(parents=True)
    ni.save(
        ni.Nifti1Image(np.random.default_rng(0).random((40, 40, 30)).astype(np.float32), AFFINE),
        source / name / "orig" / "TOF.nii.gz",
    )
    mask = np.zeros((40, 40, 30), dtype=np.uint8)
    for label, (x, y, z) in blobs:
        mask[x - 2 : x + 3, y - 2 : y + 3, z - 1 : z + 2] = label
    ni.save(ni.Nifti1Image(mask, AFFINE), source / name / "aneurysms.nii.gz")
    return mask


def test_find_cases_matches_the_pattern(tmp_path):
    for name in ("10078F", "10001", "10042B"):
        write_adam_case(tmp_path, name)
    (tmp_path / "notes").mkdir()  # no scan inside: ignored

    cases = find_cases(str(tmp_path), "{case}/orig/TOF.nii.gz")

    assert list(cases) == ["10001", "10042B", "10078F"]
    assert cases["10001"] == str(tmp_path / "10001" / "orig" / "TOF.nii.gz")


def test_find_cases_needs_a_case_placeholder(tmp_path):
    with pytest.raises(ValueError, match="{case}"):
        find_cases(str(tmp_path), "*/orig/TOF.nii.gz")


def test_aneurysm_points_describe_each_selected_component(tmp_path):
    mask = write_adam_case(tmp_path, "c", [(1, (10, 10, 10)), (1, (30, 30, 20)), (2, (10, 30, 15))])

    points = aneurysm_points(mask, AFFINE, labels=[1])

    assert points.shape == (4, 3)  # two points per untreated aneurysm
    spheres = np.hstack(
        [(points[::2] + points[1::2]) / 2, np.linalg.norm(points[::2] - points[1::2], axis=1, keepdims=True) / 2]
    )
    np.testing.assert_allclose(spheres, connected_components_to_spheres(mask == 1, AFFINE))
    assert len(aneurysm_points(mask, AFFINE, labels=None)) == 6  # any non-zero value


def run_prepare(source, data_dir, *args):
    return subprocess.run(
        [sys.executable, "-m", "deepaneseg.cli.prepare", f"source_dir={source}", f"data_dir={data_dir}", *args],
        capture_output=True,
        text=True,
    )


def test_prepare_converts_an_adam_style_dataset(tmp_path):
    source, data_dir = tmp_path / "adam", tmp_path / "data"
    mask = write_adam_case(source, "10002", [(1, (15, 15, 12)), (2, (30, 30, 20))])
    write_adam_case(source, "10001")  # no aneurysm

    result = run_prepare(source, data_dir, "labels=[1]")

    assert result.returncode == 0, result.stderr[-2000:]
    patients = fetch_patient_dirs(str(data_dir))
    assert [p.split("/")[-1] for p in patients] == ["P0001", "P0002"]
    cases = pd.read_csv(data_dir / "cases.csv")
    assert cases[["patient", "case", "aneurysms"]].values.tolist() == [["P0001", 10001, 0], ["P0002", 10002, 1]]
    assert read_aneurysm_spheres(patients[0]).shape == (0, 4)
    np.testing.assert_allclose(read_aneurysm_spheres(patients[1]), connected_components_to_spheres(mask == 1, AFFINE))
    patient = read_patient_data_base([patients[1]])[0]
    np.testing.assert_allclose(patient["affine"], AFFINE)
    assert patient["data"].shape == (40, 40, 30)


def test_prepare_refuses_to_overwrite_patients(tmp_path):
    source, data_dir = tmp_path / "adam", tmp_path / "data"
    write_adam_case(source, "10001")
    run_prepare(source, data_dir)

    result = run_prepare(source, data_dir)

    assert result.returncode != 0 and "overwrite=true" in result.stderr


def test_prepare_reports_missing_masks(tmp_path):
    source = tmp_path / "adam"
    write_adam_case(source, "10001")
    (source / "10001" / "aneurysms.nii.gz").unlink()

    result = run_prepare(source, tmp_path / "data")

    assert result.returncode != 0 and "10001" in result.stderr
