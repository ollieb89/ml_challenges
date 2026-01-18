# Project Organization and Path Fixes

## Overview
Organized the root directory by moving model files and performance reports into a structured `data/` hierarchy and updated scripts to use these paths consistently.

## Directory Structure
- **Models**: `data/models/` - Centralized storage for `.pt`, `.onnx`, and `.engine` files.
- **Reports**: `data/reports/` - Centralized storage for JSON reports and profiling HTML outputs.

## Key Changes

### 1. Model Resolution
- **`projects/pose_analyzer/src/pose_analyzer/yolo_detector.py`**: Updated `YOLOPoseDetector` to resolve `data/models/` relative to the file location. Added fallback to CWD for flexibility.
- **`projects/pose_analyzer/src/pose_analyzer/pose_detector.py`**: Updated `YOLOPosev11Detector` with similar robust path resolution for both `.pt` and `.engine` weights.
- **`projects/pose_analyzer/scripts/profile_pose_detector.py`**: Updated to load models from `data/models/` and save profiling results to `data/reports/`.

### 2. Performance Reporting & JSON Fixes
- **`projects/pose_analyzer/src/pose_analyzer/performance_tests.py`**:
    - Updated `save_report` to output to `data/reports/kalman_filter_performance_report.json`.
    - **NumpyEncoder Implementation**: Added a custom `json.JSONEncoder` subclass to handle NumPy types (`np.bool_`, `np.float64`, etc.). This fixed a `TypeError` during report serialization.
    - Explicitly cast metrics to Python native types in several test methods for additional safety.

### 3. Automation Scripts
- **`scripts/download_models.py`**: Enhanced the cleanup logic to move not just `.pt` files, but also `.onnx` and `.engine` artifacts from the root directory into `data/models/`.

## Manual Cleanup Performed
- Moved YOLOv11 engine, ONNX, and PT files to `data/models/`.
- Moved `kalman_filter_performance_report.json` to `data/reports/`.
- Removed duplicated `yolo11n-pose.pt` from `projects/pose_analyzer/`.

## Verification
- `scripts/download_models.py` runs successfully and confirms the organized structure.
- `performance_tests.py` now successfully generates and saves the JSON report in the correct location.
