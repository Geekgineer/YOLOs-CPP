# API Reference

All public classes live in the `yolos` namespace. Task-specific classes are in sub-namespaces (`yolos::det`, `yolos::seg`, `yolos::pose`, `yolos::obb`, `yolos::cls`). Include the top-level header to pull everything in:

```cpp
#include "yolos/yolos.hpp"
```

---

## Core Types

Defined in `yolos/core/types.hpp` — shared across all tasks.

### `yolos::BoundingBox`

Axis-aligned bounding box (top-left origin).

```cpp
struct BoundingBox {
    int x, y;          // top-left corner
    int width, height;  // box dimensions

    float area() const noexcept;
    float iou(const BoundingBox& other) const noexcept;
};
```

### `yolos::OrientedBoundingBox`

Rotated bounding box used by the OBB detector.

```cpp
struct OrientedBoundingBox {
    float x, y;           // center point
    float width, height;  // box dimensions
    float angle;          // rotation in radians

    float area() const noexcept;
};
```

### `yolos::KeyPoint`

Single keypoint for pose estimation.

```cpp
struct KeyPoint {
    float x, y;        // image coordinates
    float confidence;  // per-keypoint score [0, 1]
};
```

---

## Object Detection — `yolos::det`

Defined in `yolos/tasks/detection.hpp`.

### `yolos::det::Detection`

```cpp
struct Detection {
    BoundingBox box;
    float       conf;     // confidence [0, 1]
    int         classId;
};
```

### `yolos::det::YOLODetector`

Main detector — supports YOLO v7, v8, v10, v11, v26, and NAS via runtime auto-detection.

```cpp
YOLODetector(
    const std::string& modelPath,   // path to .onnx model
    const std::string& labelsPath,  // path to class-names .txt file
    bool               useGPU = false,
    YOLOVersion        version = YOLOVersion::Auto
);

// Load the model from memory instead of a file (see In-Memory Model Loading)
YOLODetector(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames,  // empty = read ONNX metadata
    bool                            useGPU = false,
    YOLOVersion                     version = YOLOVersion::Auto
);

// Run detection on a BGR image
std::vector<Detection> detect(
    const cv::Mat& image,
    float confThreshold = 0.4f,
    float iouThreshold  = 0.45f
);

// Run detection on many images in one ONNX Runtime call (see Batch Inference)
std::vector<std::vector<Detection>> batchDetect(
    const std::vector<cv::Mat>& images,
    float confThreshold = 0.4f,
    float iouThreshold  = 0.45f
);

// Draw boxes + labels onto the image
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const;

// Draw boxes with a semi-transparent filled mask
void drawDetectionsWithMask(cv::Mat& image, const std::vector<Detection>& detections,
                            float alpha = 0.4f) const;

const std::vector<std::string>& getClassNames()  const;
const std::vector<cv::Scalar>&  getClassColors() const;
```

**Version-pinned convenience classes** (same API as `YOLODetector`):

| Class | YOLO version |
|---|---|
| `yolos::det::YOLOv7Detector` | v7 |
| `yolos::det::YOLOv8Detector` | v8 |
| `yolos::det::YOLOv10Detector` | v10 |
| `yolos::det::YOLOv11Detector` | v11 |
| `yolos::det::YOLO26Detector` | v26 |
| `yolos::det::YOLONASDetector` | NAS |

**Factory functions:**

```cpp
std::unique_ptr<YOLODetector> yolos::det::createDetector(
    const std::string& modelPath,
    const std::string& labelsPath,
    YOLOVersion        version = YOLOVersion::Auto,
    bool               useGPU = false
);

std::unique_ptr<YOLODetector> yolos::det::createDetectorFromMemory(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames,
    YOLOVersion                     version = YOLOVersion::Auto,
    bool                            useGPU = false
);
```

**Example:**

```cpp
yolos::det::YOLODetector detector("yolo11n.onnx", "coco.names");
auto results = detector.detect(frame);
detector.drawDetections(frame, results);
```

---

## Instance Segmentation — `yolos::seg`

Defined in `yolos/tasks/segmentation.hpp`.

### `yolos::seg::Segmentation`

```cpp
struct Segmentation {
    BoundingBox box;
    float       conf;
    int         classId;
    cv::Mat     mask;  // binary mask (CV_8UC1), original image coords
};
```

### `yolos::seg::YOLOSegDetector`

Supports YOLOv8-seg, YOLOv11-seg, and YOLO26-seg.

```cpp
YOLOSegDetector(
    const std::string& modelPath,
    const std::string& labelsPath,
    bool               useGPU = false
);

// Load the model from memory instead of a file (see In-Memory Model Loading)
YOLOSegDetector(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames,  // empty = read ONNX metadata
    bool                            useGPU = false
);

// Returns detections with per-instance binary masks
std::vector<Segmentation> segment(
    const cv::Mat& image,
    float confThreshold = 0.4f,
    float iouThreshold  = 0.45f
);

// Segment many images in one ONNX Runtime call (see Batch Inference)
std::vector<std::vector<Segmentation>> batchSegment(
    const std::vector<cv::Mat>& images,
    float confThreshold = 0.4f,
    float iouThreshold  = 0.45f
);

// Draw masks + boxes onto the image
void drawSegmentations(cv::Mat& image, const std::vector<Segmentation>& results,
                       float maskAlpha = 0.5f) const;

void drawMasksOnly(cv::Mat& image, const std::vector<Segmentation>& results,
                   float maskAlpha = 0.5f) const;
```

**Example:**

```cpp
yolos::seg::YOLOSegDetector detector("yolo11n-seg.onnx", "coco.names");
auto results = detector.segment(frame);
detector.drawSegmentations(frame, results);
```

---

## Pose Estimation — `yolos::pose`

Defined in `yolos/tasks/pose.hpp`.

### `yolos::pose::PoseResult`

```cpp
struct PoseResult {
    BoundingBox          box;
    float                conf;
    int                  classId;           // 0 = person
    std::vector<KeyPoint> keypoints;        // 17 keypoints (COCO format)
};
```

### `yolos::pose::YOLOPoseDetector`

Supports YOLOv8-pose, YOLOv11-pose, and YOLO26-pose.

```cpp
YOLOPoseDetector(
    const std::string& modelPath,
    const std::string& labelsPath = "",  // optional; defaults to "person"
    bool               useGPU = false
);

// Load the model from memory instead of a file (see In-Memory Model Loading)
YOLOPoseDetector(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames = {},  // empty = metadata, then "person"
    bool                            useGPU = false
);

std::vector<PoseResult> detect(
    const cv::Mat& image,
    float confThreshold = 0.4f,
    float iouThreshold  = 0.5f
);

// Detect on many images in one ONNX Runtime call (see Batch Inference)
std::vector<std::vector<PoseResult>> batchDetect(
    const std::vector<cv::Mat>& images,
    float confThreshold = 0.4f,
    float iouThreshold  = 0.5f
);

// Draw bounding boxes + skeleton keypoints
void drawPoses(cv::Mat& image, const std::vector<PoseResult>& results,
               int kptRadius = 4, float kptThreshold = 0.5f, int lineThickness = 2) const;

// Draw skeleton only (no boxes)
void drawSkeletonsOnly(cv::Mat& image, const std::vector<PoseResult>& results,
                       int kptRadius = 4, float kptThreshold = 0.5f, int lineThickness = 2) const;
```

**Example:**

```cpp
yolos::pose::YOLOPoseDetector detector("yolo11n-pose.onnx");
auto results = detector.detect(frame);
detector.drawPoses(frame, results);
```

---

## Oriented Bounding Box Detection — `yolos::obb`

Defined in `yolos/tasks/obb.hpp`.

### `yolos::obb::OBBResult`

```cpp
struct OBBResult {
    OrientedBoundingBox box;
    float               conf;
    int                 classId;
};
```

### `yolos::obb::YOLOOBBDetector`

Supports YOLOv8-obb, YOLOv11-obb, and YOLO26-obb.

```cpp
YOLOOBBDetector(
    const std::string& modelPath,
    const std::string& labelsPath,
    bool               useGPU = false
);

// Load the model from memory instead of a file (see In-Memory Model Loading)
YOLOOBBDetector(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames,  // empty = read ONNX metadata
    bool                            useGPU = false
);

std::vector<OBBResult> detect(
    const cv::Mat& image,
    float confThreshold = 0.25f,
    float iouThreshold  = 0.45f,
    int   maxDet = 300
);

// Detect on many images in one ONNX Runtime call (see Batch Inference)
std::vector<std::vector<OBBResult>> batchDetect(
    const std::vector<cv::Mat>& images,
    float confThreshold = 0.25f,
    float iouThreshold  = 0.45f,
    int   maxDet = 300
);

void drawDetections(cv::Mat& image, const std::vector<OBBResult>& results,
                    int thickness = 2) const;
```

**Example:**

```cpp
yolos::obb::YOLOOBBDetector detector("yolo11n-obb.onnx", "dota.names");
auto results = detector.detect(frame);
detector.drawDetections(frame, results);
```

---

## Image Classification — `yolos::cls`

Defined in `yolos/tasks/classification.hpp`.

### `yolos::cls::ClassificationResult`

```cpp
struct ClassificationResult {
    int         classId;
    float       confidence;
    std::string className;
};
```

### `yolos::cls::YOLOClassifier`

Supports YOLOv11-cls, YOLOv12-cls, and YOLO26-cls.

```cpp
YOLOClassifier(
    const std::string& modelPath,
    const std::string& labelsPath,
    bool               useGPU = false,
    const cv::Size&    targetInputShape = cv::Size(224, 224)
);

// Load the model from memory instead of a file (see In-Memory Model Loading)
YOLOClassifier(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames,
    bool                            useGPU = false,
    const cv::Size&                 targetInputShape = cv::Size(224, 224)
);

ClassificationResult classify(const cv::Mat& image);

// Classify many images in one ONNX Runtime call (see Batch Inference)
std::vector<ClassificationResult> batchClassify(const std::vector<cv::Mat>& images);

void drawResult(cv::Mat& image, const ClassificationResult& result,
                const cv::Point& position = cv::Point(10, 30)) const;

const std::vector<std::string>& getClassNames() const;
```

**Version-pinned convenience classes:**

| Class | YOLO version |
|---|---|
| `yolos::cls::YOLO11Classifier` | v11 |
| `yolos::cls::YOLO12Classifier` | v12 |
| `yolos::cls::YOLO26Classifier` | v26 |

**Factory functions:**

```cpp
std::unique_ptr<YOLOClassifier> yolos::cls::createClassifier(
    const std::string& modelPath,
    const std::string& labelsPath,
    YOLOVersion        version = YOLOVersion::V11,
    bool               useGPU = false
);

std::unique_ptr<YOLOClassifier> yolos::cls::createClassifierFromMemory(
    const void*                     modelData,
    size_t                          modelSize,
    const std::vector<std::string>& classNames,
    YOLOVersion                     version = YOLOVersion::V11,
    bool                            useGPU = false
);
```

**Example:**

```cpp
yolos::cls::YOLOClassifier classifier("yolo11n-cls.onnx", "imagenet.names");
auto result = classifier.classify(frame);
classifier.drawResult(frame, result);
```

**Preprocessing.** `classify()` reproduces `ultralytics.data.augment.classify_transforms()`:
resize the shortest edge to the model input size, centre-crop to a square, scale to
`[0, 1]`. Ultralytics resizes through PIL, whose bilinear filter is **antialiased**, so
YOLOs-CPP uses `preprocessing::resizeAntialiasBilinear()` rather than
`cv::resize(INTER_LINEAR)`. Using a plain bilinear resize here inflates confidences and
can change the top-1 class on downscaled images.

This differs from the other tasks on purpose: Ultralytics' `LetterBox` (detection,
segmentation, pose, OBB) really does use `cv2.INTER_LINEAR`, so those paths keep it.

---

## Depth Estimation — `yolos::depth`

Defined in `yolos/tasks/depth.hpp`. Requires a YOLO26 `-depth` model.

### `yolos::depth::YOLODepthEstimator`

```cpp
YOLODepthEstimator(
    const std::string& modelPath,
    bool               useGPU = false
);

// Per-pixel metric depth in meters, CV_32FC1, sized to `image`
cv::Mat estimate(const cv::Mat& image);

// Blend a colorized depth map over the image
void drawDepth(cv::Mat& image, const cv::Mat& depth, float alpha = 0.6f,
               drawing::DepthColormap cmap = drawing::DepthColormap::Jet,
               drawing::DepthNorm mode = drawing::DepthNorm::Disparity) const;
```

No `labelsPath`, no confidence and no IoU threshold: depth models emit a single dense
map and have no classes. The constructor throws `std::runtime_error` if the model's
output is not shaped `[1, 1, H, W]`, so pointing it at a detection export fails
immediately rather than producing meaningless depth.

**Factory function:**

```cpp
std::unique_ptr<YOLODepthEstimator> yolos::depth::createDepthEstimator(
    const std::string& modelPath,
    bool               useGPU = false
);
```

**Example:**

```cpp
yolos::depth::YOLODepthEstimator estimator("yolo26n-depth.onnx");
cv::Mat depth = estimator.estimate(frame);      // CV_32FC1, meters

float metres = depth.at<float>(y, x);
double lo, hi;
cv::minMaxLoc(depth, &lo, &hi);

estimator.drawDepth(frame, depth);
```

**Units.** Values are **metric depth in meters**, straight from the model. The exported
graph already applies the `clamp`/`exp`, the log-affine calibration and the 4× upsample
from Ultralytics' `Depth` head, so nothing needs undoing on this side. The library never
normalizes depth — only the visualization helpers do.

### Depth visualization — `yolos::drawing`

```cpp
enum class DepthColormap { Jet, Inferno };
enum class DepthNorm { Disparity, Metric };

cv::Mat colorizeDepth(const cv::Mat& depth,
                      DepthColormap cmap = DepthColormap::Jet,
                      DepthNorm mode = DepthNorm::Disparity,
                      float vmin = NaN, float vmax = NaN);

void drawDepthMap(cv::Mat& image, const cv::Mat& depth, float alpha = 0.6f,
                  DepthColormap cmap = DepthColormap::Jet,
                  DepthNorm mode = DepthNorm::Disparity,
                  float vmin = NaN, float vmax = NaN);
```

Defaults match Ultralytics: disparity (`1/d`) normalization between the 2nd and 98th
percentile, JET colormap, non-positive pixels black, `alpha = 0.6`.

**For video, pass explicit `vmin`/`vmax`.** Recomputing percentiles per frame makes the
overlay flicker; the video and camera examples derive the range from the first frame and
reuse it. Ultralytics' third colormap, `spectral`, is a custom matplotlib LUT with no
OpenCV equivalent and is not supported.

**Exporting a depth model:**

```python
from ultralytics import YOLO
YOLO("yolo26n-depth.pt").export(format="onnx")
```

---

## Base Session — `yolos::OrtSessionBase`

Defined in `yolos/core/session_base.hpp`. All detectors inherit from this — you normally don't need to use it directly.

```cpp
cv::Size    getInputShape()       const noexcept;
bool        isDynamicInputShape() const noexcept;
bool        isDynamicBatchSize()  const noexcept;  // true if the ONNX batch dim is dynamic
int         getModelBatchSize()   const noexcept;  // fixed batch size, or -1 if dynamic
bool        supportsBatchSize(size_t count) const noexcept;  // can one call take `count` images?
std::string getDevice()           const noexcept;  // "cpu" or "gpu"
size_t      getNumInputNodes()    const noexcept;
size_t      getNumOutputNodes()   const noexcept;
```

---

## In-Memory Model Loading

Every task class has a constructor that takes the serialized ONNX bytes instead of a
file path, for encrypted stores, network streams and resources embedded in the binary.
Class names are passed as a `std::vector<std::string>` in class-id order, so no labels
file is needed either.

```cpp
// Whatever produces the bytes — decryption, download, embedded array — only the
// buffer reaches the detector.
std::vector<uint8_t> bytes = yolos::utils::readFileBytes("yolo11n.onnx");

yolos::det::YOLODetector detector(
    bytes.data(), bytes.size(),
    {"person", "bicycle", "car" /* ... */});

auto results = detector.detect(frame);
```

Notes:

- ONNX Runtime **copies** the buffer while creating the session, so the caller may
  free or wipe it as soon as the constructor returns.
- Passing an empty `classNames` falls back to the Ultralytics `names` entry in the
  ONNX metadata, when the export carries one.
- A null pointer or zero size throws `std::invalid_argument`.
- `yolos::utils::readFileBytes(path)` is a convenience helper for testing this path;
  production callers normally supply their own bytes.

Constructors and matching `create*FromMemory()` factories exist for detection,
segmentation, pose, OBB, classification and YOLOE.

---

## Batch Inference

`batchDetect` / `batchSegment` / `batchClassify` pack several images into a single
`N × C × H × W` tensor and run one ONNX Runtime call, which is what improves GPU
throughput over calling the single-image method in a loop.

```cpp
std::vector<cv::Mat> images = {cv::imread("a.jpg"), cv::imread("b.jpg"), cv::imread("c.jpg")};

yolos::det::YOLODetector detector("yolo11n.onnx", "coco.names", /*useGPU=*/true);

// One result vector per input image, in input order
std::vector<std::vector<yolos::det::Detection>> results = detector.batchDetect(images);
```

| Task | Method | Returns |
|---|---|---|
| Detection | `batchDetect(images, conf, iou)` | `std::vector<std::vector<Detection>>` |
| Segmentation | `batchSegment(images, conf, iou)` | `std::vector<std::vector<Segmentation>>` |
| Pose | `batchDetect(images, conf, iou)` | `std::vector<std::vector<PoseResult>>` |
| OBB | `batchDetect(images, conf, iou, maxDet)` | `std::vector<std::vector<OBBResult>>` |
| Classification | `batchClassify(images)` | `std::vector<ClassificationResult>` |

**Automatic fallback.** True batching needs a model whose batch dimension accepts the
batch size — either a dynamic batch dim, or a fixed one equal to `images.size()`. When
it does not, the batch methods loop over the single-image path instead, so the call
still returns one result per image with any export. The same recovery covers exports
that *declare* a dynamic batch dimension the graph cannot actually run (hard-coded
`Reshape` targets, for example): the failure is reported as a warning and the per-image
loop takes over. Check up front with:

```cpp
detector.isDynamicBatchSize();               // was the model exported with dynamic=True?
detector.getModelBatchSize();                // fixed batch size, or -1 when dynamic
detector.supportsBatchSize(images.size());   // will this call actually batch?
```

Export a dynamic-batch model from Ultralytics with:

```python
model.export(format="onnx", dynamic=True)
```

**Behavioral note.** A batched tensor requires one shared letterbox target, so batched
runs letterbox every image to the model input shape. For a model with a *dynamic input
shape*, the single-image methods instead pick a per-image stride-aligned shape, so the
two paths can differ slightly for such models. Fixed-input-shape models — the common
case — produce identical results either way.

Runnable demo: `src/batch_image_inference.cpp` (`--in-memory` also exercises the
in-memory loader).

---

## YOLO Version Enum

```cpp
enum class YOLOVersion {
    Auto,  // runtime detection (default)
    V7, V8, V10, V11, V12, V26, NAS
};
```
