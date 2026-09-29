"""Experimental code that is not part of the published (MICCAI 2021) pipeline and is not maintained.

- models: Proposition1/2/3, 3D U-Net variants coupled with vessel segmentation or deep supervision.
  Selectable in get_model by their names. Known issue: Proposition2 applies the sigmoid twice to vessel outputs.
- trainer_dual: trainer for models taking a (patch, vessel patch) pair; only covered by import-level checks.

The matching analysis scripts are in scripts/experimental.
"""
