import torch, random
import numpy as np

from torch.utils.data import DataLoader
from utils import get_logger

from Data.Generators import generateTransforms
from Data.IO import points_to_spheres
import Volume.Patch as vp

logger = get_logger("Data Preparation")
def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, patches, size, dim, neg_trans, neg_rot, neg_disp, pos_trans, pos_rot, pos_disp, flip, noise):
        random.shuffle(patches)
        self.patches = patches
        self.size = size
        self.dim = dim
        self.neg_trans = neg_trans
        self.neg_rot = neg_rot
        self.neg_disp = neg_disp
        self.pos_trans = pos_trans
        self.pos_rot = pos_rot
        self.pos_disp = pos_disp
        self.flip = flip
        self.noise = noise


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.patches)

    def __getitem__(self, index):
        'Generates one sample of data (Data, Truth)'
        p = self.patches[index]
        if p['status'] == False:
            trans, rot, disp = self.neg_trans, self.neg_rot, self.neg_disp
        else:
            trans, rot, disp = self.pos_trans, self.pos_rot, self.pos_disp
        affine, disp = generateTransforms(trans, rot, p['point'], disp)
        v, t, _, v2m = vp.getPatchAndTruth(p['data'], p['vox2met'], p['point'], self.size, self.dim, p['aneurysms'], affine=affine, disp=disp) #, flip=self.flip, noise=self.noise)
        return  torch.from_numpy(v[np.newaxis]).float(), torch.from_numpy(t[np.newaxis]).float()

#        v_large, _, vessel, _ = vp.getPatchAndTruth(p['data'], p['vox2met'], p['point'], [x * 2 for x in self.size], self.dim, p['aneurysms'], affine=affine, disp=disp, flip=self.flip, noise=self.noise,vessel=p['vessel'])
#        return torch.from_numpy(v[np.newaxis]).float(), torch.from_numpy(v_large[np.newaxis]).float(), torch.from_numpy(t[np.newaxis]).float(), torch.from_numpy(vessel[np.newaxis]).float()

def getPatches(pat_db, pos_dup=50, batch_size=8):
    p_list = [{'point': p,'data': d['data'], 'vessel': d['vessel'], 'vox2met':d['affine'], 'aneurysms':d['aneurysms'], 'status':False} for d in pat_db for p in d['points']]
    a_list = [{'point': p,'data': d['data'], 'vessel': d['vessel'], 'vox2met':d['affine'], 'aneurysms':d['aneurysms'], 'status':True} for d in pat_db for p in points_to_spheres(d['aneurysms'])[:,:3]]
    patches = p_list + a_list * pos_dup
    return patches, get_number_of_steps(len(patches), batch_size)

def getDataloaders(train_db, valid_db, config, workers=4, shuffle_train=True, shuffle_val=False):
    train_generator, train_iterations = None, 0
    if train_db is not None:
        train_patches, train_iterations = getPatches(train_db, config['positive duplicates'], config["batch_size"])
#        train_patches, train_iterations = getPatches(train_db, 1, config["batch_size"])
        train_dataset = Dataset(train_patches, config["patch_size"], config["patch_shape"], config["negative sample shift"], config["negative sample rotation"], config["negative sample distortion"], config["positive sample shift"], config["positive sample rotation"], config["positive sample distortion"], config["flip"], config["noise"])
        train_generator = DataLoader(train_dataset, batch_size= config["batch_size"], shuffle = shuffle_train, num_workers = workers, pin_memory = True)
        logger.info(f"Training DataLoader is created: (patches: {len(train_patches)}, batch size: {config['batch_size']}, shuffle: {shuffle_train}, workers: {workers})")

    valid_generator, valid_iterations = None, 0
    if valid_db is not None:
        valid_patches, valid_iterations = getPatches(valid_db, config['positive duplicates'], config["validation_batch_size"])
        valid_dataset = Dataset(valid_patches, config["patch_size"], config["patch_shape"], config["negative sample shift"], config["negative sample rotation"], config["negative sample distortion"], config["positive sample shift"], config["positive sample rotation"], config["positive sample distortion"], config["flip"], config["noise"])
#        valid_patches, valid_iterations = getPatches(valid_db, 1, config["validation_batch_size"])
#        valid_dataset = Dataset(valid_patches, config["patch_size"], config["patch_shape"], None, None, None, None, None, None, None, None)
        valid_generator = DataLoader(valid_dataset, batch_size = config["validation_batch_size"], shuffle = shuffle_val, num_workers = workers, pin_memory = True)
        logger.info(f"Validation DataLoader is created: (patches: {len(valid_patches)}, batch size: {config['validation_batch_size']}, shuffle: {shuffle_val}, workers: {workers})")

    return {"train": train_generator, "valid": valid_generator}, {'train' : train_iterations, 'valid' : valid_iterations}
