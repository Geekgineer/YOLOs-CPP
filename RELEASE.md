# Release Guide for YOLOs-CPP

This guide explains how to prepare and publish a release for YOLOs-CPP.

## Pre-Release Checklist

### 1. Verify All Tests Pass

```bash
cd tests
./test_all.sh
```

Expected output: All 36 tests should pass (100%).

### 2. Run Benchmarks (Optional)

```bash
cd benchmarks
./auto_bench.sh
```

### 3. Update Version Numbers

Update version in:
- `benchmarks/yolo_unified_benchmark.cpp` (line ~54: `BENCHMARK_VERSION`)
- `README.md` (badges and documentation)

### 4. Prepare Model Assets

The test scripts need pre-trained models. To avoid depending on external repos, host models in your own releases:

```bash
./scripts/prepare_release.sh
```

This creates zip files in `release_assets/`:
- `yolo-detection-models.zip`
- `yolo-segmentation-models.zip`
- `yolo-pose-models.zip`
- `yolo-obb-models.zip`
- `yolo-classification-models.zip`

## Creating the Release

### Step 1: Create Model Assets Release

First, create a release for model assets (this only needs to be done once, or when adding new models):

1. Go to GitHub → Releases → "Create new release"
2. **Tag:** `v1.0.0-models`
3. **Title:** "Model Assets v1.0.0"
4. **Description:**
   ```
   Pre-trained YOLO models for YOLOs-CPP tests.
   
   Included models:
   - Detection: YOLOv5, v6, v8, v9, v10, v11, v12, YOLO26
   - Segmentation: YOLOv8, v11, YOLO26
   - Pose: YOLOv8, v11, YOLO26
   - OBB: YOLOv8, v11, YOLO26
   - Classification: YOLOv8, v11, YOLO26
   ```
5. Upload all `.zip` files from `release_assets/`
6. Publish release

### Step 2: Update Download Scripts

After creating the models release, update the download scripts to use your release tag:

```bash
# In each tests/*/models/download_test_models.sh
RELEASE_TAG="v1.0.0-models"  # Update this to your actual tag
```

### Step 3: Create Main Release

1. Go to GitHub → Releases → "Create new release"
2. **Tag:** `v1.0.0` (follow semantic versioning)
3. **Title:** "YOLOs-CPP v1.0.0"
4. **Description:** Use the template below
5. Generate release notes automatically or write manually
6. Publish release

## Release Notes Template

```markdown
## What's New in v1.0.0

### Features
- Support for YOLO26 (end-to-end NMS-free architecture)
- All 5 task types: Detection, Segmentation, Pose, OBB, Classification
- Comprehensive benchmark suite
- 100% test coverage (36/36 tests passing)

### Supported Models
| Version | Detection | Segmentation | Pose | OBB | Classification |
|---------|-----------|--------------|------|-----|----------------|
| YOLOv5  | ✅ | - | - | - | - |
| YOLOv6  | ✅ | - | - | - | - |
| YOLOv8  | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv9  | ✅ | - | - | - | - |
| YOLOv10 | ✅ | - | - | - | - |
| YOLOv11 | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv12 | ✅ | - | - | - | - |
| YOLO26  | ✅ | ✅ | ✅ | ✅ | ✅ |

### Performance (CPU - Intel i7-1185G7)
| Model | FPS | Latency (avg) |
|-------|-----|---------------|
| YOLOv11n | 97 | 10.3 ms |
| YOLOv8n | 86 | 11.7 ms |
| YOLO26n | 78 | 12.8 ms |

### Requirements
- CMake 3.16+
- C++17 compiler
- OpenCV 4.5+
- ONNX Runtime 1.16+

### Breaking Changes
- `test/` renamed to `tests/`
- `benchmark_unified/` renamed to `benchmarks/`

### Contributors
- @Geekgineer
```

## Post-Release

### Verify CI/CD

After release, verify that:
1. GitHub Actions workflow passes
2. Tests can download models from your releases
3. Documentation links work

### Update Documentation

Update any external documentation or wikis to reflect the new version.

## Troubleshooting

### Tests fail to download models

If the download scripts fail:
1. Verify the release tag exists
2. Check that all zip files are attached to the release
3. Ensure the release is public

### CI/CD fails

Check the workflow logs for:
1. ONNX Runtime download failures (network issues)
2. Build errors (missing dependencies)
3. Test failures (model compatibility)

## File Locations

| Purpose | Location |
|---------|----------|
| Download scripts | `tests/*/models/download_test_models.sh` |
| CI/CD workflow | `.github/workflows/main.yml` |
| Release prep script | `scripts/prepare_release.sh` |
| Model assets output | `release_assets/` |
