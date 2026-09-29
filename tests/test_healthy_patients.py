"""Patients without aneurysm (healthy controls) must be usable for training and evaluation."""

import numpy as np

from deepaneseg.inference.evaluation import adam_evaluation, confusion_matrix
from deepaneseg.training.dataset import get_patches

NO_SPHERES = np.empty((0, 4))


def patient(aneurysms):
    return {"data": None, "vessel": None, "affine": np.eye(4), "aneurysms": aneurysms, "points": np.zeros((3, 3))}


def test_healthy_patient_contributes_only_negative_patches():
    ill = patient(np.array([[0.0, 0, 0], [2.0, 0, 0]]))

    patches, _ = get_patches([patient(None), ill], pos_dup=2)

    assert sum(p["status"] for p in patches) == 2  # the one aneurysm, duplicated twice
    assert sum(not p["status"] for p in patches) == 6  # 3 negative points per patient


def test_confusion_matrix_counts_detections_on_healthy_patient_as_false_positives():
    pred = np.zeros((20, 20, 20))
    pred[2:5, 2:5, 2:5] = 1
    pred[12:15, 12:15, 12:15] = 1

    tp, fn, fp, cmat = confusion_matrix(pred, np.eye(4), NO_SPHERES)

    assert (tp, fn, fp) == (0, 0, 2)
    assert cmat.tolist() == [[0, 54], [0, 20**3 - 54]]


def test_confusion_matrix_on_healthy_patient_without_detection():
    tp, fn, fp, _ = confusion_matrix(np.zeros((10, 10, 10)), np.eye(4), NO_SPHERES)

    assert (tp, fn, fp) == (0, 0, 0)


def test_adam_evaluation_on_healthy_patient():
    pred = np.zeros((20, 20, 20))
    pred[2:5, 2:5, 2:5] = 1

    tp, fp, _, _ = adam_evaluation(pred, np.eye(4), NO_SPHERES)

    assert (tp, fp) == (0, 1)


def test_points_are_extracted_for_a_patient_without_aneurysm(tmp_path):
    import json

    import nibabel as ni
    import pandas as pd

    from deepaneseg.data.io import extract_points_from_patient

    vol = np.random.default_rng(0).random((40, 40, 40)).astype(np.float32)
    ni.save(ni.Nifti1Image(vol, np.diag([0.5, 0.5, 0.5, 1.0])), tmp_path / "noskull.nii.gz")
    (tmp_path / "config.json").write_text(json.dumps({"noskull volume": "noskull.nii.gz"}))

    extract_points_from_patient(str(tmp_path), r=4, nb_points=5)

    points = pd.read_csv(tmp_path / "points.csv")
    assert set(points["type"]) == {"Vessel", "Parenchyma"} and len(points) == 10
