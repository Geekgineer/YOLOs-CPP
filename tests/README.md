# YOLOs-CPP Test Suite

Comprehensive test suite validating C++ YOLO implementations against Python Ultralytics reference.

## Test Status

**Parity** tests compare C++ output against a fresh Ultralytics Python run on the
same weights and images. **Self-contained** tests assert library behaviour against
synthetic ONNX models or fixed reference values — no downloaded weights and no
Ultralytics reference run.

| Task | Parity | Self-contained | Total | Models | Status |
|------|-------:|---------------:|------:|--------|--------|
| Detection | 7 | 3 (letterbox geometry) | 10 | YOLOv5, v6, v8, v9, v10, v11, v12, YOLO26 | ✅ Pass |
| Classification | 6 | 7 (Pillow-parity resize) | 13 | YOLOv8, v11, YOLO26 | ✅ Pass |
| Pose | 7 | — | 7 | YOLOv8, v11, YOLO26 | ✅ Pass |
| Segmentation | 8 | — | 8 | YOLOv8, v11, YOLO26 | ✅ Pass |
| OBB | 7 | — | 7 | YOLOv8, v11, YOLO26 | ✅ Pass |
| YOLOE | 8 | — | 8 | yoloe-26n-seg (open-vocab, export + ONNX parity) | ✅ Pass |
| API (batch + in-memory) | — | 22 | 22 | synthetic ONNX (no weights needed) | ✅ Pass |
| Depth | 7 | 25 (postprocessing + synthetic) | 32 | yolo26n-depth (metric depth, dense-map parity) | ✅ Pass |
| **Total** | **50** | **57** | **107** | | **100%** |

Counts are gtest cases, read off `--gtest_list_tests` of the built binaries.

Detection, classification and depth compile their self-contained tests into the same
`compare_*` binary as their parity tests, so both halves run in one job. The API
suite is its own binary, `test_api_batch_and_memory`, with no parity half.

Of the 57 self-contained tests, 27 need nothing but the compiler and can be run
directly after a build:

```bash
cd build
./compare_detection_results     --gtest_filter='LetterboxConsistency.*'   #  3
./compare_classification_results --gtest_filter='AntialiasResize.*'        #  7
./compare_depth_results --gtest_filter='CropLetterboxAndResize.*:ColorizeDepth.*:DrawDepthMap.*'  # 17
```

The remaining 30 — the 22 API tests and the 8 `SyntheticDepthTest` cases — need
Python to generate their synthetic ONNX models first (`make_synthetic_models.py`),
but never run Ultralytics inference.

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
./test_api.sh
./test_depth.sh

# Build only the YOLOE parity suite (after Python reference exists under yoloe/results/)
./build_test.sh 6

# Build only the API suite (batch inference + in-memory loading)
./build_test.sh 7
```

## How Tests Work

The parity suites (detection, classification, segmentation, pose, OBB, YOLOE) compare
against Ultralytics:

1. **Model Download**: Downloads pretrained `.pt` files from Ultralytics
2. **ONNX Export**: Exports models to ONNX format (opset 12)
3. **Python Inference**: Runs Ultralytics to generate ground truth
4. **C++ Build**: Builds C++ inference executables
5. **C++ Inference**: Runs C++ implementation
6. **Comparison**: Compares results using GoogleTest

The **API suite** (`test_api.sh`) is different: it tests library behavior rather than
numeric parity, so it generates tiny synthetic ONNX models with
`api/make_synthetic_models.py` (needs only `onnx` and `numpy`) and asserts that

- `batchDetect` / `batchSegment` / `batchClassify` return exactly what a per-image loop returns
- each batch slice is postprocessed against its own image, not against batch element 0
- fixed-batch exports — and exports that declare a batch dim their graph cannot
  actually run — transparently fall back to the per-image loop
- loading a model from memory behaves identically to loading it from disk

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
├── test_api.sh             # Batch inference + in-memory loading (no weights needed)
├── build_test.sh           # CMake build script
├── CMakeLists.txt          # Test suite CMake config
├── api/
│   ├── make_synthetic_models.py    # Generates tiny ONNX models (shapes only)
│   ├── test_batch_and_memory.cpp
│   └── models/                     # Synthetic .onnx files (generated)
│
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
├── test_depth.sh           # Depth estimation parity + self-contained tests
├── depth/
│   ├── make_synthetic_models.py     # Tiny ONNX models, no weights needed
│   ├── inference_depth_cpp.cpp
│   ├── inference_depth_ultralytics.py
│   ├── compare_results.cpp          # Per-pixel AbsRel / delta-1 parity
│   ├── test_postprocessing.cpp      # Crop, rescale and colormap unit tests
│   ├── inference_config.json
│   ├── models/
│   └── results/                     # JSON index + raw float32 maps (generated)
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
| Depth mean AbsRel | 1e-4 | Mean relative depth error vs Ultralytics (measured: ~1.8e-06) |
| Depth max relative error | 1e-3 | Worst-pixel relative depth error vs Ultralytics (measured: ~5.2e-06) |
| Depth range | 1e-3 | Agreement on the map's min and max depth |
| Depth δ1 | 0.99 | Fraction of pixels within 1% of the reference (measured: 1.00) |

## CI/CD Integration

The test scripts are designed for CI/CD pipelines:

- Uses `uv` for fast, reproducible Python environment
- Auto-downloads ONNX Runtime for the platform
- Exports models with compatible opset (12)
- Returns proper exit codes (0 = pass, non-zero = fail)

**GitHub Actions** (`.github/workflows/main.yml`) runs each task in parallel: `detection`, `segmentation`, `pose`, `obb`, `classification`, **`yoloe`** (`tests/test_yoloe.sh`), **`api`** (`tests/test_api.sh`), and **`depth`** (`tests/test_depth.sh`). Artifacts upload `tests/<task>/results/` per matrix job, excluding depth's raw `.bin` maps.

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
5. **Classification preprocessing**: `inference_classification_ultralytics.py` calls
   `ultralytics.data.augment.classify_transforms()` for the reference tensor, so the
   comparison is against Ultralytics itself. It previously re-implemented the C++ OpenCV
   preprocessing, which meant the suite compared YOLOs-CPP against a Python transcription
   of itself and could not detect a mismatch (issue #137). `classification/test_preprocessing.cpp`
   additionally pins the antialiased resize to golden Pillow output, so it is guarded even
   without a Python run.
4. **YOLOE**: `tests/yoloe/inference_config.json` lists the same `classes` as `models/export_yoloe_test_onnx.py` and C++ `YOLOESegDetector`. Python reference uses the exported ONNX (no `set_classes` on ONNX). Inference enumerates images in **sorted** order so JSON matches C++. C++ tests bundle ONNX Runtime 1.20.x; Ultralytics may install a different Python `onnxruntime` for ONNX inference—outputs should still match within tolerances.
5. **Depth**: the reference comes from Ultralytics' own `DepthPredictor`, so its
   `LetterBox` preprocessing and `ops.scale_masks` postprocessing are ground truth.
   Measured agreement on `yolo26n-depth` is mean AbsRel ~1.8e-06 with 100% of pixels
   inside the 1% band, i.e. effectively bit-level. The suite checks both a mean and a
   max relative error: a one-pixel letterbox-crop shift (the failure mode a rounding
   mistake produces) yields a mean AbsRel of only ~9.9e-04 — the mean averages the
   defect away over ~100k pixels — while the max relative error jumps to ~1.8e-02.
   Thresholds sit ~55x (mean) and ~190x (max) above that real residual rather than at
   the measured value, so cross-version drift is not flaky: the two sides run different
   ONNX Runtime builds, and `cv::resize` and `torch.nn.functional.interpolate` are
   independent bilinear implementations. Dense maps are written as raw `float32` with a
   JSON index and compared per pixel — summary statistics alone would pass a map that is
   wrong everywhere but has the right range. `make_synthetic_models.py` also builds a
   constant-depth ONNX, which needs no weights and is the strictest check of the
   crop-and-rescale path.

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
