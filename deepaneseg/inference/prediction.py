import os
import math

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
import deepaneseg.data.io as dio


def _to_sitk(data, spacing, origin=(0.0, 0.0, 0.0)):
    """numpy volume indexed (x, y, z) -> SimpleITK image (which stores arrays as (z, y, x))"""
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(data)))
    image.SetSpacing([float(s) for s in spacing])
    image.SetOrigin([float(o) for o in origin])
    return image


def _from_sitk(image):
    return np.transpose(sitk.GetArrayFromImage(image))


def _resample(image, shape, spacing, origin, interpolation, default_value):
    interpolators = {"linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor}
    if interpolation not in interpolators:
        raise ValueError(f"'interpolation' must be one of {sorted(interpolators)}, got '{interpolation}'")
    reference = sitk.Image([int(n) for n in shape], image.GetPixelID())
    reference.SetSpacing([float(s) for s in spacing])
    reference.SetOrigin([float(o) for o in origin])
    return _from_sitk(sitk.Resample(image, reference, sitk.Transform(), interpolators[interpolation], default_value))


def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.0):
    """
    Resamples data (voxel size spacing, in mm) to voxel size target_spacing, covering the same field of view:
    the outer corners of the first voxels coincide. Undo with resample_to_grid.
    """
    spacing, target_spacing = np.asarray(spacing, float), np.asarray(target_spacing, float)
    shape = np.ceil(np.round(np.asarray(data.shape) * spacing / target_spacing, 5)).astype(int)
    origin = (target_spacing - spacing) / 2  # in the frame where data's first voxel is centred at 0
    return _resample(_to_sitk(data, spacing), shape, target_spacing, origin, interpolation, default_value)


def resample_to_grid(data, spacing, target_shape, target_spacing, interpolation="linear", default_value=0.0):
    """
    Inverse of resample_to_spacing: resamples data (voxel size spacing) back onto the original grid of shape
    target_shape and voxel size target_spacing, so that the result can be saved with the original affine.
    """
    spacing, target_spacing = np.asarray(spacing, float), np.asarray(target_spacing, float)
    origin = (spacing - target_spacing) / 2  # data's first voxel centre, in the target grid's frame
    return _resample(
        _to_sitk(data, spacing, origin), target_shape, target_spacing, (0, 0, 0), interpolation, default_value
    )


def patch_wise_prediction(device, model, data, patch_shape, margin=0, batch_size=1, patient_name=""):
    """
    Predicts a whole volume with a model working on patches of patch_shape voxels.
    Patches overlap by 2 * margin voxels; only their central part (without margin) is kept. The volume is padded
    (with its minimum value) so that every voxel, including the borders, is predicted exactly once.
    """
    patch_shape = np.asarray(patch_shape, dtype=int)
    margin = np.broadcast_to(np.asarray(margin, dtype=int), patch_shape.shape)
    core = patch_shape - 2 * margin
    if np.any(core <= 0):
        raise ValueError(f"margin {margin.tolist()} leaves no voxel to predict in patches of {patch_shape.tolist()}")

    image_shape = np.asarray(data.shape)
    n_tiles = np.ceil(image_shape / core).astype(int)
    pad_after = n_tiles * core - image_shape + margin
    padded = np.pad(data, list(zip(margin, pad_after)), mode="constant", constant_values=data.min())
    output = np.zeros(n_tiles * core)

    starts = np.mgrid[: n_tiles[0], : n_tiles[1], : n_tiles[2]].reshape((3, -1)).T * core
    loop = tqdm(range(0, len(starts), batch_size), leave=True, unit="batch", total=math.ceil(len(starts) / batch_size))
    loop.set_description(f"Prediction: {patient_name}")
    for j in loop:
        idx = starts[j : j + batch_size]
        batch = np.stack([padded[(np.newaxis,) + tuple(slice(s, s + p) for s, p in zip(i, patch_shape))] for i in idx])
        prediction = model(torch.from_numpy(batch).float().to(device)).detach().cpu().numpy()
        for i, p in zip(idx, prediction):
            output[tuple(slice(s, s + c) for s, c in zip(i, core))] = p[0][
                tuple(slice(m, m + c) for m, c in zip(margin, core))
            ]

    return output[tuple(slice(0, n) for n in image_shape)]


def predict_patient(pat_dict, device, model, patch_size, patch_shape, output_dir, margin=0, batch_size=1):
    """
    Predicts one patient and writes the (sigmoid) prediction, on the patient's voxel grid, to
    <output_dir>/<patient>.nii.gz.
    :param pat_dict: patient data, as returned by data.io.read_patient_data_base (including the 'dir' key)
    :param patch_size: size in mm of a patch, and patch_shape its size in voxels (as used for training)
    """
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(pat_dict["dir"])
    affine, data = pat_dict["affine"], pat_dict["data"]
    print(f"Processing {name}")

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    work_spacing = np.asarray(patch_size, float) / np.asarray(patch_shape)
    work = resample_to_spacing(data, spacing, work_spacing)
    prediction = patch_wise_prediction(
        device, model, work, patch_shape, margin=margin, batch_size=batch_size, patient_name=name
    )
    prediction = resample_to_grid(prediction, work_spacing, data.shape, spacing)

    nib.Nifti1Image(prediction, affine).to_filename(os.path.join(output_dir, name + ".nii.gz"))


@torch.no_grad()
def predict_patients(pat_db, device, model, patch_size, patch_shape, output_dir=".", margin=0, batch_size=1):
    for p in pat_db:
        predict_patient(p, device, model, patch_size, patch_shape, output_dir, margin=margin, batch_size=batch_size)


@torch.no_grad()
def get_patient_prediction(
    pat_path, device, model, patch_size, patch_shape=(48, 48, 48), margin=8, batch_size=10, normalization="Normal"
):
    """Predicts one patient directory; returns the prediction (on the patient's grid), its affine and the
    ground-truth aneurysm spheres."""
    pat_dict = dio.read_patient_data_base([pat_path], volume="init volume", normalize=normalization)[0]
    affine, data = pat_dict["affine"], pat_dict["data"]

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    work_spacing = np.asarray(patch_size, float) / np.asarray(patch_shape)
    work = resample_to_spacing(data, spacing, work_spacing)
    prediction = patch_wise_prediction(
        device,
        model,
        work,
        patch_shape,
        margin=margin,
        batch_size=batch_size,
        patient_name=os.path.basename(pat_dict["dir"]),
    )
    prediction = resample_to_grid(prediction, work_spacing, data.shape, spacing)

    spheres = dio.points_to_spheres(pat_dict["aneurysms"])
    return prediction, affine, spheres
