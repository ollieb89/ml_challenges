# Scripts Directory

This directory contains utility scripts used throughout the **ai-ml-pipeline** project.

## `vram_stress_test.py`

**Purpose**:  Determine the maximum batch size that can be processed by the YOLO PoseDetector model on an 8 GB VRAM GPU (e.g., a RTX 3070 Ti). The script incrementally increases the batch size, measures VRAM consumption, and stops when an out‑of‑memory (OOM) error occurs.

**How to run**:
```bash
# Activate the project's Pixi environment first (if not already active)
pixi shell --manifest-path .
# Execute the stress test, providing the path to the YOLO model checkpoint
python scripts/vram_stress_test.py --model data/models/yolo11n-pose.pt
```

*Optional arguments*:
- `--device <index>` – Choose a CUDA device (default 0).
- `--batches 1 2 4 8 …` – Supply a custom list of batch sizes.
- `--no-plot` – Skip generating the memory‑vs‑batch plot.

**Outputs**:
- `reports/vram_stress.csv` – CSV log of `batch_size, memory_used_MB, oom_flag`.
- `reports/vram_curve.png` – Plot of VRAM usage vs. batch size (if `matplotlib` is installed).

**Dependencies**:
- PyTorch with CUDA support (already a project dependency).
- Optional: `matplotlib` and `pandas` for plotting.

Use this script whenever you need to profile VRAM limits for new models or hardware configurations.
