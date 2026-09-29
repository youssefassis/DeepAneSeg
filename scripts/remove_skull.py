#!/usr/bin/env python3
"""Skull-strip every patient volume and register it as 'noskull volume' in the patient config.json."""

import argparse
import json
import os

import nibabel as ni
import numpy as np

import deepaneseg.data.io as dio
import deepaneseg.volume.selection as vs

OUTPUT_NAME = "noskull.nii.gz"


def remove_skull(patient_dir):
    config_file = os.path.join(patient_dir, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    ni_vol = ni.load(os.path.join(patient_dir, config["init volume"]))
    vol = np.asarray(ni_vol.dataobj).astype(np.float32)

    vol = vs.remove_skull_mask(vol) * vol
    vol /= np.max(vol)

    ni.Nifti1Image(vol, ni_vol.affine).to_filename(os.path.join(patient_dir, OUTPUT_NAME))
    config["noskull volume"] = OUTPUT_NAME
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory containing the patient folders (P????)")
    args = parser.parse_args()

    for patient_dir in dio.fetch_patient_dirs(args.data_dir):
        print(f"Removing skull: {patient_dir}")
        remove_skull(patient_dir)


if __name__ == "__main__":
    main()
