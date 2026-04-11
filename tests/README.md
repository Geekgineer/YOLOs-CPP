# YOLOs-CPP Test Suite

Comprehensive test suite validating C++ YOLO implementations against Python Ultralytics reference.

## Test Status

| Task | Tests | Models | Status |
|------|-------|--------|--------|
| Detection | 8/8 | YOLOv5, v6, v8, v9, v10, v11, v12, YOLO26 | ✅ Pass |
| Classification | 6/6 | YOLOv8, v11, YOLO26 | ✅ Pass |
| Pose | 7/7 | YOLOv8, v11, YOLO26 | ✅ Pass |
| Segmentation | 8/8 | YOLOv8, v11, YOLO26 | ✅ Pass |
| OBB | 7/7 | YOLOv8, v11, YOLO26 | ✅ Pass |
| YOLOE | 8/8 | yoloe-26n-seg (open-vocab, export + ONNX parity) | ✅ Pass |
| **Total** | **44/44** parity | | **100%** |

## Requirements

- **Python 3.10+** with `uv` package manager (auto-installed)
- **CMake 3.16+**
- **OpenCV 4.x**
- **ONNX Runtime 1.20+** (auto-downloaded)

## Quick Start

```bash
# Run all tests
./test_all.sh

# Run individual task tests
./test_detection.sh
./test_classification.sh
./test_pose.sh
./test_segmentation.sh
./test_obb.sh
./test_yoloe.sh

# Build only the YOLOE parity suite (after Python reference exists under yoloe/results/)
./build_test.sh 6
```

## How Tests Work

1. **Model Download**: Downloads pretrained `.pt` files from Ultralytics
2. **ONNX Export**: Exports models to ONNX format (opset 12)
3. **Python Inference**: Runs Ultralytics to generate ground truth
4. **C++ Build**: Builds C++ inference executables
5. **C++ Inference**: Runs C++ implementation
6. **Comparison**: Compares results using GoogleTest

## Directory Structure

```
tests/
├── test_utils.sh           # Shared utilities (uv, venv, exports)
├── test_all.sh             # Master test runner
├── test_detection.sh       # Detection task runner
├── test_classification.sh  # Classification task runner
├── test_segmentation.sh    # Segmentation task runner
├── test_pose.sh            # Pose estimation task runner
├── test_obb.sh             # OBB detection task runner
├── test_yoloe.sh           # YOLOE open-vocabulary segmentation parity
├── build_test.sh           # CMake build script
├── CMakeLists.txt          # Test suite CMake config
├── yoloe/
│   ├── inference_config.json       # conf, iou, and `classes` (must match export)
│   ├── inference_yoloe_cpp.cpp
│   ├── inference_yoloe_ultralytics.py
│   ├── compare_results.cpp
│   ├── models/
│   │   └── export_yoloe_test_onnx.py   # Exports yoloe-26n-seg.onnx for tests
│   ├── data/images/
│   └── results/                      # JSON + masks (generated)
│
├── detection/
│   ├── models/             # .pt and .onnx models
│   ├── data/images/        # Test images
│   ├── results/            # JSON results
│   ├── inference_detection_cpp.cpp
│   ├── inference_detection_ultralytics.py
│   └── compare_results.cpp
│
├── classification/         # Similar structure
├── segmentation/           # Similar structure
├── pose/                   # Similar structure
└── obb/                    # Similar structure
```

## Tolerance Settings

The comparison tests use configurable error margins:

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| Confidence | ±0.2 | Accounts for preprocessing differences |
| Bounding Box | ±50px | Pixel coordinate tolerance |
| Keypoints | ±20px | Pose keypoint position tolerance |
| Mask Pixels | 10% | Segmentation mask difference (invalid pixel ratio) |
| OBB Center | ±50px | Oriented box center tolerance |
| OBB Angle | ±0.2 rad | Rotation angle tolerance |

## CI/CD Integration

The test scripts are designed for CI/CD pipelines:

- Uses `uv` for fast, reproducible Python environment
- Auto-downloads ONNX Runtime for the platform
- Exports models with compatible opset (12)
- Returns proper exit codes (0 = pass, non-zero = fail)

**GitHub Actions** (`.github/workflows/main.yml`) runs each task in parallel: `detection`, `segmentation`, `pose`, `obb`, `classification`, and **`yoloe`** (`tests/test_yoloe.sh`). Artifacts upload `tests/<task>/results/` per matrix job.

```yaml
# Example: run full suite locally (same tasks as CI matrix combined)
- name: Run YOLOs-CPP Tests
  run: |
    cd tests
    ./test_all.sh
```

## Notes

1. **Model size**: Uses smaller input (320x320) for faster testing
2. **YOLO26 models**: Feature end-to-end NMS-free architecture
3. **VOC dataset**: Detection models are fine-tuned on Pascal VOC (20 classes)
4. **YOLOE**: `tests/yoloe/inference_config.json` lists the same `classes` as `models/export_yoloe_test_onnx.py` and C++ `YOLOESegDetector`. Python reference uses the exported ONNX (no `set_classes` on ONNX). Inference enumerates images in **sorted** order so JSON matches C++. C++ tests bundle ONNX Runtime 1.20.x; Ultralytics may install a different Python `onnxruntime` for ONNX inference—outputs should still match within tolerances.

## Troubleshooting

**Python package issues:**
```bash
# Manually install packages
source ~/.yolos-cpp-test-venv/bin/activate
uv pip install ultralytics onnx tqdm
```

**ONNX Runtime errors:**
```bash
# Clear and re-download
rm -rf onnxruntime-*
./build_test.sh <task_id>
```

**Model export fails:**
```bash
# Export manually with opset 12
python3 -c "from ultralytics import YOLO; YOLO('model.pt').export(format='onnx', opset=12)"
```
