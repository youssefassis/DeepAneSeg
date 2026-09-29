#!/usr/bin/env python3
"""Skull-strip every patient volume; configuration in configs/remove_skull.yaml.

Example: python scripts/remove_skull.py data_dir=/data
The result is saved next to each volume and registered as 'noskull volume' in the patient's config.json.
"""

import json
import os

import hydra
import nibabel as ni
import numpy as np

import deepaneseg.data.io as dio
import deepaneseg.volume.selection as vs


def remove_skull(patient_dir, output_name):
    config_file = os.path.join(patient_dir, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    ni_vol = ni.load(os.path.join(patient_dir, config["init volume"]))
    vol = vs.skull_strip(np.asarray(ni_vol.dataobj))

    ni.Nifti1Image(vol, ni_vol.affine).to_filename(os.path.join(patient_dir, output_name))
    config["noskull volume"] = output_name
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


@hydra.main(version_base="1.3", config_path="../configs", config_name="remove_skull")
def main(cfg):
    for patient_dir in dio.fetch_patient_dirs(cfg.data.data_dir):
        print(f"Removing skull: {patient_dir}")
        remove_skull(patient_dir, cfg.output)


if __name__ == "__main__":
    main()
