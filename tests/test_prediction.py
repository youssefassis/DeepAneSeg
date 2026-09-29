import numpy as np
import pytest
import scipy.ndimage as ndi
import torch

from deepaneseg.inference.prediction import patch_wise_prediction, resample_to_grid, resample_to_spacing


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("shape", [(110, 100, 80), (30, 50, 20), (48, 48, 48)])
def test_patch_wise_prediction_covers_every_voxel_in_place(shape):
    data = np.random.default_rng(0).random(shape)

    output = patch_wise_prediction("cpu", Identity(), data, patch_shape=(48, 48, 48), margin=8, batch_size=4)

    np.testing.assert_allclose(output, data, rtol=1e-6)  # float32 through the model


def test_patch_wise_prediction_uses_the_given_patch_shape():
    seen = []

    class Recorder(Identity):
        def forward(self, x):
            seen.append(tuple(x.shape[-3:]))
            return x

    patch_wise_prediction("cpu", Recorder(), np.zeros((40, 40, 40)), patch_shape=(32, 24, 16), margin=4)

    assert set(seen) == {(32, 24, 16)}


def test_resampling_round_trip_restores_the_original_grid():
    spacing, work_spacing = np.array([0.4, 0.4, 0.6]), np.full(3, 38 / 48)
    vol = np.zeros((203, 187, 97))
    vol[120, 60, 30] = 1
    vol = ndi.gaussian_filter(vol, 6)  # smooth blob, well resolved at both resolutions

    work = resample_to_spacing(vol, spacing, work_spacing)
    back = resample_to_grid(work, work_spacing, vol.shape, spacing)

    assert back.shape == vol.shape
    assert np.unravel_index(np.argmax(back), back.shape) == (120, 60, 30)
    np.testing.assert_allclose(back, vol, atol=0.05 * vol.max())


def test_resample_to_spacing_keeps_physical_positions():
    spacing, work_spacing = np.array([0.5, 0.5, 0.5]), np.array([1.0, 1.0, 1.0])
    vol = np.zeros((40, 40, 40))
    vol[21:23, 9:11, 29:31] = 1  # 1 mm cube centred at voxels (21.5, 9.5, 29.5), i.e. at (10.75, 4.75, 14.75) mm

    work = resample_to_spacing(vol, spacing, work_spacing)

    # work voxel i is centred at (i + 0.5) * 1 mm - 0.25 mm: 10.75 mm -> i = 10.5, so the cube spans voxels 10-11
    assert work.shape == (20, 20, 20)
    assert work[10:12, 4:6, 14:16].sum() == pytest.approx(vol.sum() / 8, rel=0.3)
