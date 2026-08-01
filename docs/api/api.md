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

// Run detection on a BGR image
std::vector<Detection> detect(
    const cv::Mat& image,
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

**Factory function:**

```cpp
std::unique_ptr<YOLODetector> yolos::det::createDetector(
    const std::string& modelPath,
    const std::string& labelsPath,
    YOLOVersion        version = YOLOVersion::Auto,
    bool               useGPU = false
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

// Returns detections with per-instance binary masks
std::vector<Segmentation> segment(
    const cv::Mat& image,
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

std::vector<PoseResult> detect(
    const cv::Mat& image,
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

std::vector<OBBResult> detect(
    const cv::Mat& image,
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

ClassificationResult classify(const cv::Mat& image);

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

**Factory function:**

```cpp
std::unique_ptr<YOLOClassifier> yolos::cls::createClassifier(
    const std::string& modelPath,
    const std::string& labelsPath,
    YOLOVersion        version = YOLOVersion::V11,
    bool               useGPU = false
);
```

**Example:**

```cpp
yolos::cls::YOLOClassifier classifier("yolo11n-cls.onnx", "imagenet.names");
auto result = classifier.classify(frame);
classifier.drawResult(frame, result);
```

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
cv::Size    getInputShape()     const noexcept;
bool        isDynamicInputShape() const noexcept;
std::string getDevice()         const noexcept;  // "cpu" or "gpu"
size_t      getNumInputNodes()  const noexcept;
size_t      getNumOutputNodes() const noexcept;
```

---

## YOLO Version Enum

```cpp
enum class YOLOVersion {
    Auto,  // runtime detection (default)
    V7, V8, V10, V11, V12, V26, NAS
};
```
