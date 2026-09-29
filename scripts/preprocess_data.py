#!/usr/bin/env python3
"""Split the patients into training/validation/testing sets and write the base configuration.

Outputs <data_dir>/0Work/BaseConfig/ndl_config.json and split_pats.json.
"""
import argparse
import json
import os
import sys

import deepaneseg.data.generators as dgen
import deepaneseg.data.io as dio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', help='directory containing the patient folders (P????)')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.2, 0.1], metavar=('TRAIN', 'VALID', 'TEST'),
                        help='portions of patients used for training, validation and testing')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing split')
    args = parser.parse_args()

    config = dict()
    config['base_dir'] = os.path.abspath(args.data_dir)
    config['working_dir'] = os.path.join(config['base_dir'], '0Work', 'BaseConfig')
    config["labels"] = (1,)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    config["nb_channels"] = 1
    config["truth_channel"] = config["nb_channels"]
    config["data_split"] = args.split
    config["split_file"] = os.path.join(config['working_dir'], "split_pats.json")

    if os.path.isfile(config["split_file"]) and not args.overwrite:
        sys.exit(f"{config['split_file']} already exists; pass --overwrite to replace it")

    os.makedirs(config['working_dir'], exist_ok=True)
    with open(os.path.join(config['working_dir'], 'ndl_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    pat_list = dio.fetch_patient_dirs(config['base_dir'])
    dio.save_split(config['split_file'], *dgen.split_pat_list(pat_list, *config['data_split']))


if __name__ == '__main__':
    main()
