# Experimental analysis scripts

Unmaintained scripts from the PhD experiments. They contain hardcoded cluster paths and patient lists, and must be
edited before use:

- `eval.py`: ADAM challenge metrics (sensitivity, false positive count) on a folder of predictions.
- `evaluate_perf.py`: voxel-, patch- and aneurysm-wise metrics on predictions (older variant of `eval.py`).
- `evaluate.py`: patch-wise evaluation ported from TensorFlow; does not run in its current state.

The ADAM evaluation logic itself lives in `deepaneseg/inference/evaluation.py`.
