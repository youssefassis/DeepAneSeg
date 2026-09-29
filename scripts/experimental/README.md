# Experimental analysis scripts

Analysis scripts from the PhD experiments. Paths and patient lists are set at the top of each script; adapt them
before use:

- `eval.py`: ADAM challenge metrics (sensitivity, false positive count) on a folder of predictions.
- `evaluate_perf.py`: voxel-, patch- and aneurysm-wise metrics on predictions (older variant of `eval.py`).
- `evaluate.py`: patch-wise evaluation from an earlier TensorFlow version; needs porting to PyTorch before use.

`scripts/evaluate.py` computes the ADAM metrics from the training configuration, without editing.
