# Release Guide for YOLOs-CPP

This guide explains how to prepare and publish a release for YOLOs-CPP.

## Pre-Release Checklist

### 1. Verify All Tests Pass

```bash
cd tests
./test_all.sh
```

Expected output: all eight suites pass — 107 tests total (50 Ultralytics-parity,
57 self-contained). CI runs the same eight suites as a matrix, one job per task.

### 2. Run Benchmarks (Optional)

```bash
cd benchmarks
./auto_bench.sh
```

### 3. Update Version Numbers

Update the library version in:
- `CMakeLists.txt` (line 5: `project(YOLOs-CPP VERSION ...)`)
- `tests/CMakeLists.txt` (line 5: `project(yolos_cpp_tests VERSION ...)`)

Leave `benchmarks/` alone unless the benchmark tool itself changed —
`BENCHMARK_VERSION` in `benchmarks/yolo_unified_benchmark.cpp` versions the
benchmark suite, not the library, and the two are deliberately independent.

The README release badge is `shields.io/github/v/release`, which reads the latest
tag from the GitHub API. It needs no edit; it updates itself once the tag is pushed.

Do update the README **Latest News** list and, if the test counts changed, the
**Testing** table — those are hand-maintained.

### 4. Prepare Model Assets

Most parity suites need pre-trained weights. Where they come from differs by task,
and only the first group needs a release asset:

| Task | Source |
|------|--------|
| Detection, Segmentation, Pose, OBB, Classification | `v1.0.0-models` release assets, via `tests/*/models/download_test_models.sh` |
| Depth | Ultralytics assets directly (`YOLO('yolo26n-depth.pt')`) |
| YOLOE | Exported on the fly by `tests/yoloe/models/export_yoloe_test_onnx.py` |
| API (batch + in-memory) | Synthetic ONNX built at test time by `tests/api/models/make_synthetic_models.py` |

So a new model-assets release is only needed when the **detection, segmentation,
pose, OBB or classification** weight sets change. To build one:

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

Start from GitHub's generated notes so every contributor in the range is credited,
then edit them into the shape below:

```bash
gh release create vX.Y.Z --title "YOLOs-CPP vX.Y.Z" --generate-notes --draft
```

Review the draft, replace the body with the template below, then publish.

Pick the version with semantic versioning applied to the **public headers** in
`include/`. The check that decides major vs minor:

```bash
git diff <previous-tag>..HEAD -- include/
```

A removed or re-signatured public declaration is a major bump — including in the
low-level `yolos::preprocessing` and `yolos::core` utilities, not just the task
classes. Purely additive constructors and methods are a minor bump. Note that a
change to output *values* (preprocessing, coordinate math) is not an API break, but
it does belong under **Behavior changes** below, because it can move a downstream
user's numbers without any compile error to warn them.

## Release Notes Template

```markdown
## What's New in vX.Y.Z

### Features
- <new tasks, new APIs, new model support>

### Behavior changes
- <changes that move output values without breaking compilation — preprocessing
  fixes, coordinate math, threshold defaults. Say what moved and by how much.>

### Breaking changes
- <removed or re-signatured public declarations, with the old and new signature>

### Supported Models
<copy the table from README.md — it is the single source of truth>

### Performance
<copy from README.md "Benchmarks", or regenerate with benchmarks/auto_bench.sh.
Keep the device label attached to each number: the README figures are RTX 3060
GPU except where the row says CPU.>

### Testing
- <N> tests across <M> suites (<parity> Ultralytics-parity, <self> self-contained)

### Requirements
- CMake 3.16+
- C++17 compiler
- OpenCV 4.5+
- ONNX Runtime 1.16+

### Contributors
<keep the list gh --generate-notes produced; it catches external contributors>
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
