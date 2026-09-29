#!/usr/bin/env python3
"""Split the patients into training/validation/testing sets; configuration in configs/preprocess.yaml.

Example: python scripts/preprocess_data.py data_dir=/data split=[0.7,0.2,0.1]
Writes the split to data.split_file (<data_dir>/0Work/split_pats.json by default).
"""

import os
import sys

import hydra

import deepaneseg.data.generators as dgen
import deepaneseg.data.io as dio


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess")
def main(cfg):
    split_file = cfg.data.split_file
    if os.path.isfile(split_file) and not cfg.overwrite:
        sys.exit(f"{split_file} already exists; pass overwrite=true to replace it")

    os.makedirs(os.path.dirname(os.path.abspath(split_file)), exist_ok=True)
    pat_list = dio.fetch_patient_dirs(os.path.abspath(cfg.data.data_dir))
    dio.save_split(split_file, *dgen.split_pat_list(pat_list, *cfg.split))


if __name__ == "__main__":
    main()
