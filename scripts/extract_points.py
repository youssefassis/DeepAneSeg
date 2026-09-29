#!/usr/bin/env python3
"""Select negative patch centers (on vessels and in the parenchyma) for every patient.

Writes <output> (CSV) and its 3D Slicer .fcsv counterpart in each patient folder.
Requires the 'noskull volume' produced by remove_skull.py.
"""

import argparse

import deepaneseg.data.io as dio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="directory containing the patient folders (P????)")
    parser.add_argument("--radius", type=float, default=20, help="minimum distance between two points, in mm")
    parser.add_argument("--n-points", type=int, default=100, help="maximum number of points of each type")
    parser.add_argument("--output", default="points.csv", help="output file name, inside each patient folder")
    parser.add_argument(
        "--random", action="store_true", help="pick random parenchyma points instead of vessel + parenchyma"
    )
    args = parser.parse_args()

    dio.extract_points(
        args.data_dir, r=args.radius, nb_points=args.n_points, outfile=args.output, random_points=args.random
    )


if __name__ == "__main__":
    main()
