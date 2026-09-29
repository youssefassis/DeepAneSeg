#!/usr/bin/env python3
"""Select negative patch centers (on vessels and in the parenchyma) for every patient.

Configuration in configs/extract_points.yaml. Example: python scripts/extract_points.py data_dir=/data
Writes <output> (CSV) and its 3D Slicer .fcsv counterpart in each patient folder.
Requires the 'noskull volume' produced by remove_skull.py.
"""

import hydra

import deepaneseg.data.io as dio


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract_points")
def main(cfg):
    dio.extract_points(
        cfg.data.data_dir, r=cfg.radius, nb_points=cfg.n_points, outfile=cfg.output, random_points=cfg.random
    )


if __name__ == "__main__":
    main()
