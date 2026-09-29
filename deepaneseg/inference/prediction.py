import os
import math

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
import deepaneseg.data.io as dio


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing) / 2


def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.0):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(
        size=new_size,
        spacing=new_spacing,
        direction=image.GetDirection(),
        origin=image.GetOrigin() + offset,
        default_value=default_value,
    )
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def sitk_resample_to_image(
    image, reference_image, default_value=0.0, interpolator=sitk.sitkLinear, transform=None, output_pixel_type=None
):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.0):
    image = sitk.GetImageFromArray(np.ones(size, dtype=float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.0):
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation == "linear":
        interpolator = sitk.sitkLinear
    elif interpolation == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError(
            "'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(interpolation)
        )
    resampled_image = sitk_resample_to_spacing(
        image, new_spacing=target_spacing, interpolator=interpolator, default_value=default_value
    )
    return sitk_image_to_data(resampled_image)


def data_to_sitk_image(data, spacing=(1.0, 1.0, 1.0)):
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=float))
    return image


def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0, 2))
    return data


def ndl_patch_wise_prediction(device, model, data, margin=0, batch_size=1, patient_name=""):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = [48, 48, 48]  # list(map(int,model.input.shape[-3:]))
    image_shape = np.array(data.shape)
    # compute indices
    if isinstance(margin, int):
        margin = np.asarray([margin] * len(image_shape))
    delta = np.array(patch_shape) - 2 * margin
    n_idx = np.floor((image_shape - 2 * margin) / delta).astype(np.int64)
    overflow = image_shape - (n_idx * delta + 2 * margin)
    start = overflow // 2
    indices = np.mgrid[: n_idx[0], : n_idx[1], : n_idx[2]].reshape((3, -1)).T
    indices = indices * delta + start
    output = np.zeros(data.shape)

    print(f"{indices.shape[0]} patches to process")
    loop = tqdm(
        range(0, indices.shape[0], batch_size),
        leave=True,
        unit="batch",
        total=int(math.ceil(indices.shape[0] / batch_size)),
    )
    for j in loop:  # range(0,indices.shape[0],batch_size):
        k = min(j + batch_size, len(indices))
        idx = indices[j:k]
        batch = np.asarray(
            [
                data[
                    np.newaxis, i[0] : i[0] + patch_shape[0], i[1] : i[1] + patch_shape[1], i[2] : i[2] + patch_shape[2]
                ]
                for i in idx
            ]
        )
        batch = torch.from_numpy(batch).float()
        prediction = model(batch.to(device))
        for i, p in zip(idx, prediction.detach().cpu().numpy()):
            s = i + margin
            # reconstruct shape
            output[s[0] : s[0] + delta[0], s[1] : s[1] + delta[1], s[2] : s[2] + delta[2]] = p[
                0, margin[0] : margin[0] + delta[0], margin[1] : margin[1] + delta[1], margin[2] : margin[2] + delta[2]
            ]
        loop.set_description(f"Prediction: {patient_name}")

    return output


def ndl_run_validation_case(pat_dict, device, size, output_dir, model, margin=0, batch_size=1):
    """
    Runs a test case and writes predicted images to file.
    :param pat_dict: info about the patient data. Dictionary as returned by data.io.read_patient_data_base (including the 'dir' key)
    :param size: 3-tuple that provides the size in mm of a patch (patch shape will be extracted from model info)
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is
    considered a positive result and will be assigned a label.
    :param model: model to use for prediction
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = pat_dict["affine"]
    test_data = pat_dict["data"]
    output_file = os.path.join(output_dir, os.path.basename(pat_dict["dir"]) + ".nii.gz")
    print(f"Processing {os.path.basename(pat_dict['dir'])}")

    # reshape input volume
    old_spacing = np.linalg.norm(affine, axis=0)[:3]
    patch_shape = np.array([48, 48, 48])  # np.array([int(dim) for dim in model.input.shape[-3:]])
    new_spacing = np.array(size) / patch_shape
    tvol = resample_to_spacing(test_data, old_spacing, new_spacing)

    prediction = ndl_patch_wise_prediction(device, model=model, data=tvol, margin=margin, batch_size=batch_size)

    timage = resample_to_spacing(prediction, new_spacing, old_spacing)

    image = nib.Nifti1Image(timage, affine)
    image.to_filename(output_file)


@torch.no_grad()
def ndl_run_validation_cases(pat_db, device, model, patch_size, output_dir=".", margin=0, batch_size=1):
    for p in pat_db:
        ndl_run_validation_case(
            p, device, output_dir=output_dir, model=model, size=patch_size, margin=margin, batch_size=batch_size
        )


@torch.no_grad()
def get_patient_prediction(
    pat_path, device, model, patch_size, output_dir=".", margin=8, batch_size=10, normalization="Normal"
):
    pat_dict = dio.read_patient_data_base([pat_path], volume="init volume", normalize=normalization)[0]

    affine = pat_dict["affine"]
    test_data = pat_dict["data"]
    print(f"Processing {os.path.basename(pat_dict['dir'])}")

    # reshape input volume
    old_spacing = np.linalg.norm(affine, axis=0)[:3]
    patch_shape = np.array([48, 48, 48])  # np.array([int(dim) for dim in model.input.shape[-3:]])
    new_spacing = np.array(patch_size) / patch_shape
    tvol = resample_to_spacing(test_data, old_spacing, new_spacing)

    prediction = ndl_patch_wise_prediction(
        device,
        model=model,
        data=tvol,
        margin=margin,
        batch_size=batch_size,
        patient_name=os.path.basename(pat_dict["dir"]),
    )

    prediction = resample_to_spacing(prediction, new_spacing, old_spacing)

    # Get truth
    spheres = dio.points_to_spheres(pat_dict["aneurysms"])
    return prediction, affine, spheres  # cm, tp_diam, fn_diam #scores#timage, truth
