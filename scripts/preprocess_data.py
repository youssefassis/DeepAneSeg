import os
import json
import deepaneseg.data.io as dio
import deepaneseg.data.generators as dgen

config = dict()
config['base_dir'] = '/home/yassis/Data'
config['working_dir'] = os.path.join(config['base_dir'],'0Work/BaseConfig')
if not os.path.exists(config['working_dir']):
    os.mkdir(config['working_dir'])

config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["nb_channels"] = 1
config["truth_channel"] = config["nb_channels"]
config["data_split"] = [0.7, 0.2, 0.1]  # training/validation/testing portions of data

config["split_file"] = os.path.join(config['working_dir'], "split_pats.json")

if (os.path.isfile(config["split_file"]) and not config["overwrite"]):
        assert False, f"{config['split_file']} already exists !!, You can set overwrite to True"

# saving configuration as json file in working directory
with open(os.path.join(config['working_dir'], 'ndl_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

pat_list = dio.fetch_patient_dirs(config['base_dir'])
dio.saveSplit(config['split_file'], *(dgen.splitPatList(pat_list, *(config['data_split']))))