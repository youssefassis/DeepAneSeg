#!/usr/bin/env python3
import os
import sys
import json

import Data.IO as dio
import torch

from utils import get_logger, load_model, get_number_of_learnable_parameters
from prediction import ndl_run_validation_cases

def main():
    d = sys.argv[1]
    cfg = os.path.join(d,'ndl_config.json')
    try:
        with open(cfg,'r') as f:
            config = json.load(f)
    except:
        print(f'No such config file {cfg}')
        exit()

    logger = get_logger('Data Preparation')

    normalize = config['normalize'] if 'normalize' in config else None

    _, _, test_list = dio.readSplit(config['split_file'])
#    test_list = ["/home/yassis/Data/P0078"]
    logger.info(f"{len(test_list)} Patients for testing")

    volume = "noskull volume" # or init volume

    test_db = dio.read_patient_data_base(test_list, volume=volume, normalize=normalize)
    logger.info(f"'{volume}' successfully loaded with '{normalize}' normalization")

    logger = get_logger('Model')

    model = load_model(config ,logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending the model to '{device}'")

    model.eval()
    logger.info("Setting the model to evaluation mode")

    ndl_run_validation_cases(test_db,
                             device = device,
                             model = model,
                             patch_size = config['patch_size'],
                             output_dir = os.path.join(d, "prediction/test"),
                             margin = 8,
                             batch_size=config['validation_batch_size'])

if __name__ == "__main__":
    main()
