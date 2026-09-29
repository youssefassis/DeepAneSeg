import numpy as np
import pytest

from deepaneseg.data.generators import generate_transforms
from deepaneseg.data.io import points_to_spheres
from deepaneseg.volume.edition import draw_sphere, get_truth
from deepaneseg.volume.patch import get_patch, get_patch_and_truth

ISOTROPIC_HALF_MM = np.diag([0.5, 0.5, 0.5, 1.0])


def test_points_to_spheres_returns_center_and_radius():
    points = np.array([[0.0, 0, 0], [2.0, 0, 0], [10.0, 10, 10], [10.0, 10, 16]])

    spheres = points_to_spheres(points)

    np.testing.assert_allclose(spheres, [[1, 0, 0, 1], [10, 10, 13, 3]])


def test_draw_sphere_burns_voxels_within_radius_only():
    vol = np.zeros((40, 40, 40), dtype=np.uint8)

    draw_sphere(vol, ISOTROPIC_HALF_MM, c=[10.0, 10.0, 10.0], r=2.0, val=1)

    idx = np.argwhere(vol == 1)
    distances = np.linalg.norm(idx * 0.5 - 10.0, axis=1)
    assert distances.max() <= 2.0
    assert vol[20, 20, 20] == 1
    # a sphere of radius 2 mm at 0.5 mm resolution covers roughly 4/3*pi*4^3 voxels
    assert abs(len(idx) - 4 / 3 * np.pi * 4**3) / len(idx) < 0.1


def test_get_truth_labels_each_sphere():
    vol = np.zeros((40, 40, 40))
    spheres = np.array([[5.0, 5, 5, 1.5], [15.0, 15, 15, 1.5]])

    truth = get_truth(vol, ISOTROPIC_HALF_MM, spheres)

    assert set(np.unique(truth)) == {0, 1, 2}


def test_get_patch_without_transform_samples_the_volume():
    vol = np.random.default_rng(0).random((40, 40, 40))
    center = np.array([10.0, 10.0, 10.0])

    # odd patch of 9 voxels over 4.5 mm at 0.5 mm resolution: patch voxel centers land on volume voxels
    patch, patch_vox2met = get_patch(vol, ISOTROPIC_HALF_MM, center, size=4.5, dim=9)

    start = 16  # first patch voxel center: 10 - 4.5/2 + 0.25 = 8 mm, i.e. volume voxel 16
    np.testing.assert_allclose(patch, vol[start : start + 9, start : start + 9, start : start + 9], atol=1e-12)
    np.testing.assert_allclose(np.diag(patch_vox2met)[:3], 0.5)


def test_get_patch_and_truth_marks_the_aneurysm_under_augmentation():
    np.random.seed(0)
    vol = np.random.default_rng(0).random((60, 60, 60))
    center = np.array([15.0, 15.0, 15.0])
    aneurysm = np.array([[13.0, 15, 15], [17.0, 15, 15]])  # 2 mm radius at patch center
    affine, disp = generate_transforms(trans=2, rot=30, center=center, disp=1)

    patch, truth, vessel, _ = get_patch_and_truth(vol, ISOTROPIC_HALF_MM, center, 19, 48, aneurysm, affine, disp)

    assert patch.shape == truth.shape == (48, 48, 48)
    assert vessel is None
    assert set(np.unique(truth)) == {0, 1}
    assert truth.sum() > 0


def test_generate_transforms_keeps_patch_center_fixed_under_distortion():
    np.random.seed(1)

    affine, disp = generate_transforms(trans=None, rot=None, disp=3)

    assert affine is None
    assert all(d.shape == (3, 3, 3) and d[1, 1, 1] == 0 for d in disp)
    assert all(np.abs(d).max() <= 3 for d in disp)


def test_selected_points_are_voxel_centers_consistent_with_patch_sampling():
    from deepaneseg.volume.selection import select_points

    vox2met = np.array([[0.4, 0, 0, -10.0], [0, 0.4, 0, 5.0], [0, 0, 0.6, 2.0], [0, 0, 0, 1]])
    vol = np.zeros((30, 30, 30))
    vol[12, 7, 20] = 1.0

    (point,) = select_points(vol, vox2met, thres_low=0.5, r=1)

    np.testing.assert_allclose(point, (vox2met @ [12, 7, 20, 1])[:3])
    patch, _ = get_patch(vol, vox2met, point, size=[1.2, 1.2, 1.8], dim=3)  # 3 voxels at the volume resolution
    assert patch[1, 1, 1] == pytest.approx(1.0)
