"""Conversion of a dataset of scans and aneurysm masks to the layout used by the pipeline."""

import glob
import os
import re

import numpy as np

from deepaneseg.volume.selection import connected_components_to_spheres


def find_cases(source_dir, scan_pattern):
    """
    Returns {case name: scan path}, sorted by case, for the files matching scan_pattern in source_dir.
    scan_pattern is a path relative to source_dir where {case} stands for one folder or file name part,
    e.g. "{case}/orig/TOF.nii.gz" (ADAM challenge) or "scans/{case}.nii.gz".
    """
    if scan_pattern.count("{case}") != 1:
        raise ValueError(f"scan pattern must contain {{case}} exactly once, got '{scan_pattern}'")
    before, after = scan_pattern.split("{case}")
    regex = re.compile(re.escape(before) + r"([^/]+)" + re.escape(after) + "$")
    cases = {}
    for path in glob.glob(os.path.join(source_dir, scan_pattern.replace("{case}", "*"))):
        match = regex.search(os.path.relpath(path, source_dir))
        if match:
            cases[match.group(1)] = path
    return dict(sorted(cases.items()))


def aneurysm_points(mask, affine, labels=None):
    """
    Annotation points of the aneurysms of a label mask: each connected component of the selected labels (any
    non-zero value if labels is None) is approximated by a sphere, described by the two ends of a diameter along x,
    as in the F.csv files. Returns a 2Nx3 array of points in mm.
    """
    selected = mask > 0 if labels is None else np.isin(mask, list(labels))
    spheres = connected_components_to_spheres(selected, affine)
    offset = np.zeros_like(spheres[:, :3])
    offset[:, 0] = spheres[:, 3]
    points = np.empty((2 * len(spheres), 3))
    points[0::2], points[1::2] = spheres[:, :3] - offset, spheres[:, :3] + offset
    return points
