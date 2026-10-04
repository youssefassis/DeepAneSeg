import numpy as np
import random
from scipy.spatial.transform import Rotation


def generate_transforms(trans=None, rot=None, center=None, disp=None):
    """
    generate a augmentation transform as a pair of affine,disp (see patch.get_patch function)
    trans: translation amplitude. Can be a 3-vector (along x,y,z) or a scalar (same amplitude in all directions)
    rot: rotation amplitude (in degrees). Can be a 3-vector (around x,y,z) or a scalar (same angle amplitude around all axes)
    center: if set, center of rotation. Else, center is (0,0,0)
    disp: amplitude for the non-rigid (spline) deformation. Can be a 3-vector (along x,y,z) or a scalar (same amplitude in all directions).
    Return affine, disp as a pair of transform parameters that can be used as (respectively) affine and disp arguments when calling patch.get_patch
    If both trans and rot are None, then the returned affine is None
    If disp is None, then the returned disp is None
    (in compliance with patch.get_patch default params)
    returned disp is a 3-tuple of displacements to apply on control points in the x,y and z dimensions. The shape of each tuple is (3,3,3) so that the deformation
    is controlled by a mesh of 3x3x3 control points superimposed on the patch (corners correspond). The returned disp makes sure that no displacement is
    applied on the center point of the patch (disp[:][1,1,1]=0)
    """
    if trans is None and rot is None:  # and scal is None:
        affine = None
    else:
        t = np.zeros(3)
        if trans is not None:
            if np.isscalar(trans):
                t = trans * np.ones(3)
            elif len(trans) == 3:
                t = trans
            else:
                raise TypeError("trans argument should be either a scalar or a 3-vector")
        t = t * (2 * (np.random.random_sample(3) - 0.5))
        r = np.zeros(3)
        if rot is not None:
            if np.isscalar(rot):
                r = rot * np.ones(3)
            elif len(rot) == 3:
                r = rot
            else:
                raise TypeError("rot argument should either a scalar or a 3-vector")

        r = r * (2 * (np.random.random_sample(3) - 0.5))
        affine = np.eye(4)
        affine[:3, :3] = Rotation.from_euler("zyx", r, degrees=True).as_matrix()
        affine[:3, 3] = t if center is None else t + center.ravel() @ (np.eye(3) - affine[:3, :3]).T

    if disp is not None:
        if np.isscalar(disp):
            amp = disp * np.ones(3)
        elif len(disp) == 3:
            amp = disp
        else:
            raise TypeError("disp argument should be either a scalar or a 3-vector")

        dx = amp[0] * 2 * (np.random.random_sample((3, 3, 3)) - 0.5)
        dy = amp[1] * 2 * (np.random.random_sample((3, 3, 3)) - 0.5)
        dz = amp[2] * 2 * (np.random.random_sample((3, 3, 3)) - 0.5)
        dx[1, 1, 1] = dy[1, 1, 1] = dz[1, 1, 1] = 0
        disp = (dx, dy, dz)
    return affine, disp


def split_pat_list(pat_list, training_pct=0.7, validation_pct=0.2, testing_pct=0.1, seed=None):
    """
    split a list of patient directories according to percentages (sum is normalized to 1 if necessary).
    returns three lists for (resp.) training, validation and testing
    The split only depends on the seed (and the patients), not on the order of pat_list; seed=None gives a random split.
    Example use with a dictionary list da, as returned by io.read_patient_data_base()
    train_list, valid_list, test_list = split_pat_list([d['dir'] for d in da], seed=0)
    """
    train_nb = round(len(pat_list) * training_pct / (training_pct + validation_pct + testing_pct))
    valid_nb = round((len(pat_list) - train_nb) * validation_pct / (validation_pct + testing_pct))
    patients = sorted(pat_list)
    random.Random(seed).shuffle(patients)
    return patients[:train_nb], patients[train_nb : train_nb + valid_nb], patients[train_nb + valid_nb :]
