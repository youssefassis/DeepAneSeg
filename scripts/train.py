#!/usr/bin/env python3

import sys, os, json
import torch
from torch import optim

from deepaneseg.data.io import add_points_to_patient_data, read_split, read_patient_data_base
from deepaneseg.utils import create_optimizer, create_lr_scheduler, get_model
from deepaneseg.training.dataset import get_dataloaders
from deepaneseg.training.losses import get_loss_criterion
from deepaneseg.training.metrics import get_metric

from deepaneseg.training.trainer import create_trainer

def main():
    d = sys.argv[1]
    cfg = os.path.join(d,'ndl_config.json')
    try:
        with open(cfg,'r') as f:
            config = json.load(f)
    except:
        print(f'No such config file {cfg}')
        exit()

    # Model Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, default_init="keras", device=device)

    loss_criterion = get_loss_criterion(config["loss"])
    eval_criterion = get_metric(config["metrics"])
    optimizer = create_optimizer(model, learning_rate=config["initial_learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = create_lr_scheduler (optimizer, config)

    # Data Preparation
    normalize = config['normalize'] if 'normalize' in config else None
    vessel = None

    train_list, valid_list, _ = read_split(config['split_file'])
#    train_list, valid_list = ["/srv/storage/tangram@talc-data2.nancy.grid5000.fr/yassis/Data/P0071"], ["/srv/storage/tangram@talc-data2.nancy.grid5000.fr/yassis/Data/P0071"]

    train_db = add_points_to_patient_data(read_patient_data_base( train_list, normalize=normalize, vessel=vessel, label="training"), config['negative patch centers'])
    valid_db = add_points_to_patient_data(read_patient_data_base( valid_list, normalize=normalize, vessel=vessel, label="validation"), config['negative patch centers']) if len(valid_list)>0 else None

    loaders, iterations = get_dataloaders(train_db, valid_db, config, shuffle_train=True, shuffle_val=False)

    # Trainer
    trainer = create_trainer(config, device = device, model=model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                            eval_criterion=eval_criterion, loaders=loaders,max_iterations=iterations)

    trainer.fit(deep_supervision=False)

if __name__ == '__main__':
    main()
