"""Regression tests for code paths that crashed (missing imports, removed NumPy/pandas/skimage APIs)."""

import json

import nibabel as ni
import numpy as np
import pandas as pd
import pytest
import torch

from deepaneseg.data import io as dio
from deepaneseg.experimental.models import Proposition2, ProjectExciteLayer
from deepaneseg.models.models import UNet3D
from deepaneseg.training.losses import WeightedCrossEntropyLoss
from deepaneseg.training.metrics import DiceCoefficient, Kappa
from deepaneseg.utils import get_model, load_model, save_checkpoint
from deepaneseg.volume.selection import remove_skull_mask, select_points

HALF_MM = np.diag([0.5, 0.5, 0.5, 1.0])


def test_select_points_keeps_points_at_least_r_apart():
    vol = np.random.default_rng(0).random((30, 30, 30))

    points = select_points(vol, HALF_MM, thres_low=0.9, r=3, nb_points=10)

    assert len(points) == 10
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    assert d[np.triu_indices(len(points), 1)].min() > 3


def test_remove_skull_mask_returns_binary_mask():
    vol = np.zeros((40, 40, 40))
    vol[5:35, 5:35, 5:35] = 1.0

    mask = remove_skull_mask(vol)

    assert mask.shape == vol.shape
    assert set(np.unique(mask)) <= {0, 1}


def test_get_model_unet3d_builds_the_paper_unet():
    assert isinstance(get_model({"model": "unet3d"}), UNet3D)


def test_load_model_restores_saved_weights(tmp_path):
    model = get_model({"model": "unet3d"})
    save_checkpoint({"model_state_dict": model.state_dict()}, False, str(tmp_path))
    config = {"model": "unet3d", "model_file": str(tmp_path / "last_checkpoint.pytorch")}

    restored = load_model(config)

    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_proposition2_forward_with_vessel_branch():
    x = torch.randn(1, 1, 48, 48, 48)

    detections, vessels = Proposition2(f_maps=8)(x, x)

    assert detections[-1].shape == vessels[-1].shape == x.shape


def test_project_excite_layer_keeps_shape():
    x = torch.randn(1, 4, 6, 6, 6)

    assert ProjectExciteLayer(4)(x).shape == x.shape


def test_weighted_cross_entropy_runs():
    logits = torch.randn(2, 2, 4, 4, 4)
    target = torch.randint(0, 2, (2, 4, 4, 4))

    assert WeightedCrossEntropyLoss()(logits, target).item() > 0


@pytest.mark.parametrize("metric", [DiceCoefficient, Kappa])
def test_metrics_do_not_track_gradients(metric):
    pred = torch.rand(1, 1, 4, 4, 4, requires_grad=True)

    assert not metric()(pred, (pred > 0.5).float()).requires_grad


def test_fcsv_round_trip_keeps_coordinates(tmp_path):
    csv = tmp_path / "points.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0], "type": ["Vessel", "Parenchyma"]}).to_csv(csv)

    dio.csv2fcsv(str(csv))
    csv.unlink()
    dio.fcsv2csv(str(tmp_path / "points.fcsv"))

    np.testing.assert_allclose(pd.read_csv(csv)[["x", "y", "z"]].to_numpy(dtype=float), [[1, 3, 5], [2, 4, 6]])


def test_extract_points_from_patient_writes_vessel_and_parenchyma_points(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)  # a relative patient path must not depend on the working directory
    vol = np.random.default_rng(0).random((40, 40, 40)).astype(np.float32)
    ni.save(ni.Nifti1Image(vol, HALF_MM), tmp_path / "noskull.nii.gz")
    pd.DataFrame({"x": [9.0, 11.0], "y": [10.0, 10.0], "z": [10.0, 10.0]}).to_csv(tmp_path / "F.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps({"noskull volume": "noskull.nii.gz", "pts aneurysm": "F.csv"}))

    dio.extract_points_from_patient(tmp_path.name, r=4, nb_points=5)
    dio.extract_points_from_patient(tmp_path.name, r=4, nb_points=5)  # a second patient would fail after a chdir

    points = pd.read_csv(tmp_path / "points.csv")
    assert set(points["type"]) == {"Vessel", "Parenchyma"}
    assert (tmp_path / "points.fcsv").exists()
