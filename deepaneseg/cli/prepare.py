"""Convert a dataset of scans and aneurysm masks to the pipeline's layout; configuration in deepaneseg/configs/prepare.yaml.

Example (ADAM challenge data): deepaneseg-prepare source_dir=/data/ADAM data_dir=/data/converted labels=[1]
Writes one P???? folder per case (volume, F.csv with the aneurysms, config.json) and cases.csv, which maps the
patients to the original case names.
"""

import json
import os
import shutil
import sys

import hydra
import pandas as pd

import deepaneseg.data.io as dio
from deepaneseg.data.prepare import aneurysm_points, find_cases
from deepaneseg.utils import get_logger


@hydra.main(version_base="1.3", config_path="../configs", config_name="prepare")
def main(cfg):
    logger = get_logger("Prepare")
    cases = find_cases(cfg.source_dir, cfg.scan)
    if not cases:
        sys.exit(f"No scan matches '{cfg.scan}' in {cfg.source_dir}")
    masks = {case: os.path.join(cfg.source_dir, cfg.mask.replace("{case}", case)) for case in cases} if cfg.mask else {}
    missing = sorted(case for case, path in masks.items() if not os.path.isfile(path))
    if missing:
        raise FileNotFoundError(f"No aneurysm mask '{cfg.mask}' for {', '.join(missing)}")
    if dio.fetch_patient_dirs(cfg.data_dir) and not cfg.overwrite:
        sys.exit(f"{cfg.data_dir} already contains patients; pass overwrite=true to replace them")

    rows = []
    for i, (case, scan) in enumerate(cases.items(), start=1):
        patient = os.path.join(cfg.data_dir, f"P{i:04d}")
        os.makedirs(patient, exist_ok=True)
        volume = "volume.nii.gz" if scan.endswith(".gz") else "volume.nii"
        shutil.copyfile(scan, os.path.join(patient, volume))
        config = {"init volume": volume}
        points = []
        if case in masks:
            mask, affine = dio.read_nii_from_file(masks[case])
            points = aneurysm_points(mask, affine, labels=cfg.labels)
            if len(points):
                pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(os.path.join(patient, "F.csv"), index=False)
                config["pts aneurysm"] = "F.csv"
        with open(os.path.join(patient, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        rows.append({"patient": f"P{i:04d}", "case": case, "scan": scan, "aneurysms": len(points) // 2})
        logger.info(f"{case} -> P{i:04d}: {len(points) // 2} aneurysms")

    pd.DataFrame(rows).to_csv(os.path.join(cfg.data_dir, "cases.csv"), index=False)


if __name__ == "__main__":
    main()
