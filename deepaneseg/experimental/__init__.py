"""Post-paper experiments, kept apart from the published (MICCAI 2021) pipeline.

- models: Proposition1/2/3, 3D U-Net variants coupled with vessel segmentation or deep supervision.
  Proposition2 applies the sigmoid twice to its vessel outputs.
- trainer_dual: trainer for models taking a (patch, vessel patch) pair.

The matching analysis scripts are in scripts/experimental.
"""
