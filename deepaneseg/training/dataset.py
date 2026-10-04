import math
import torch
import random
import numpy as np

from torch.utils.data import DataLoader
from deepaneseg.utils import get_logger

from deepaneseg.data.generators import generate_transforms
from deepaneseg.data.io import points_to_spheres
import deepaneseg.volume.patch as vp

logger = get_logger("Data Preparation")


def get_number_of_steps(n_samples, batch_size):
    return math.ceil(n_samples / batch_size)


class Dataset(torch.utils.data.Dataset):
    """
    Patches (Data, Truth) extracted on the fly with random augmentation.
    negative/positive: augmentation amplitudes (shift, rotation, distortion) for patches without/with an aneurysm.
    """

    def __init__(self, patches, size, dim, negative, positive):
        random.shuffle(patches)
        self.patches = patches
        self.size = size
        self.dim = dim
        self.negative = negative
        self.positive = positive

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.patches)

    def __getitem__(self, index):
        "Generates one sample of data (Data, Truth)"
        p = self.patches[index]
        aug = self.positive if p["status"] else self.negative
        affine, disp = generate_transforms(aug["shift"], aug["rotation"], p["point"], aug["distortion"])
        v, t, _, v2m = vp.get_patch_and_truth(
            p["data"], p["vox2met"], p["point"], self.size, self.dim, p["aneurysms"], affine=affine, disp=disp
        )
        return torch.from_numpy(v[np.newaxis]).float(), torch.from_numpy(t[np.newaxis]).float()


def get_patches(pat_db, pos_dup=50, batch_size=8):
    p_list = [
        {
            "point": p,
            "data": d["data"],
            "vessel": d["vessel"],
            "vox2met": d["affine"],
            "aneurysms": d["aneurysms"],
            "status": False,
        }
        for d in pat_db
        for p in d["points"]
    ]
    a_list = [
        {
            "point": p,
            "data": d["data"],
            "vessel": d["vessel"],
            "vox2met": d["affine"],
            "aneurysms": d["aneurysms"],
            "status": True,
        }
        for d in pat_db
        if d["aneurysms"] is not None  # healthy patients only provide negative patches
        for p in points_to_spheres(d["aneurysms"])[:, :3]
    ]
    patches = p_list + a_list * pos_dup
    return patches, get_number_of_steps(len(patches), batch_size)


def get_dataloaders(train_db, valid_db, cfg, shuffle_train=True, shuffle_val=False):
    """
    cfg: training configuration (see configs/train.yaml), providing data.patch_size, data.patch_shape,
    augmentation, batch_size, validation_batch_size and num_workers.
    """
    size, dim = list(cfg["data"]["patch_size"]), list(cfg["data"]["patch_shape"])
    augmentation = cfg["augmentation"]

    def make_loader(pat_db, batch_size, shuffle, label):
        if pat_db is None:
            return None, 0
        patches, iterations = get_patches(pat_db, augmentation["positive"]["duplicates"], batch_size)
        dataset = Dataset(patches, size, dim, augmentation["negative"], augmentation["positive"])
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg["num_workers"], pin_memory=True
        )
        logger.info(
            f"{label} DataLoader is created: (patches: {len(patches)}, batch size: {batch_size}, "
            f"shuffle: {shuffle}, workers: {cfg['num_workers']})"
        )
        return loader, iterations

    train_loader, train_iterations = make_loader(train_db, cfg["batch_size"], shuffle_train, "Training")
    valid_loader, valid_iterations = make_loader(valid_db, cfg["validation_batch_size"], shuffle_val, "Validation")
    return {"train": train_loader, "valid": valid_loader}, {"train": train_iterations, "valid": valid_iterations}
