# Usage Guide

Complete API reference and code examples for YOLOs-CPP.

## Quick Start

```cpp
#include "yolos/yolos.hpp"

// Initialize any detector
yolos::det::YOLODetector detector("model.onnx", "labels.txt", /*gpu=*/true);

// Run inference
auto detections = detector.detect(frame, /*conf=*/0.25f, /*iou=*/0.45f);

// Visualize
detector.drawDetections(frame, detections);
```

## Namespace Structure

| Namespace | Purpose |
|-----------|---------|
| `yolos::det::` | Object detection |
| `yolos::seg::` | Instance segmentation |
| `yolos::pose::` | Pose estimation |
| `yolos::obb::` | Oriented bounding boxes |
| `yolos::cls::` | Image classification |

## Object Detection

```cpp
#include "yolos/yolos.hpp"

yolos::det::YOLODetector detector(
    "models/yolo11n.onnx",
    "models/coco.names",
    true  // GPU
);

cv::Mat image = cv::imread("image.jpg");
auto detections = detector.detect(image, 0.25f, 0.45f);

for (const auto& det : detections) {
    std::cout << det.className << ": " << det.confidence << std::endl;
}

detector.drawDetections(image, detections);
```

## Instance Segmentation

```cpp
yolos::seg::YOLOSegDetector detector(
    "models/yolo11n-seg.onnx",
    "models/coco.names",
    true
);

auto segments = detector.segment(image, 0.25f, 0.45f);
detector.drawSegmentations(image, segments, 0.5f);  // 50% opacity
```

## Pose Estimation

```cpp
yolos::pose::YOLOPoseDetector detector(
    "models/yolo11n-pose.onnx",
    "",  // No labels needed
    true
);

auto poses = detector.detect(image, 0.25f, 0.45f);
detector.drawPoses(image, poses);
```

## Oriented Bounding Boxes

```cpp
yolos::obb::YOLOOBBDetector detector(
    "models/yolo11n-obb.onnx",
    "models/Dota.names",
    true
);

auto boxes = detector.detect(image, 0.25f, 0.45f);
detector.drawOBBs(image, boxes);
```

## Image Classification

```cpp
yolos::cls::YOLOClassifier classifier(
    "models/yolo11n-cls.onnx",
    "models/imagenet_classes.txt",
    true
);

auto result = classifier.classify(image);
std::cout << result.className << ": " << result.confidence * 100 << "%" << std::endl;
```

## Depth Estimation

```cpp
#include "yolos/yolos.hpp"

yolos::depth::YOLODepthEstimator estimator("yolo26n-depth.onnx", /*gpu=*/true);

cv::Mat depth = estimator.estimate(frame);   // CV_32FC1, meters

// Read a distance directly
std::cout << depth.at<float>(frame.rows / 2, frame.cols / 2) << " m" << std::endl;

// Visualize
estimator.drawDepth(frame, depth);
```

Depth needs no labels file and has no confidence threshold — the model emits one dense
map. See [Depth Estimation](../api/api.md#depth-estimation-yolosdepth) for the units and
the colormap options.

For video, pin the colour range so the overlay does not flicker frame to frame:

```cpp
double lo, hi;
cv::minMaxLoc(firstDepth, &lo, &hi);
const float vmin = 1.0f / hi;   // disparity: near objects have the largest 1/d
const float vmax = 1.0f / lo;

yolos::drawing::drawDepthMap(frame, depth, 0.6f,
                             yolos::drawing::DepthColormap::Jet,
                             yolos::drawing::DepthNorm::Disparity, vmin, vmax);
```

## Video Processing

```cpp
cv::VideoCapture cap("video.mp4");
cv::Mat frame;

while (cap.read(frame)) {
    auto detections = detector.detect(frame);
    detector.drawDetections(frame, detections);
    cv::imshow("Detection", frame);
    if (cv::waitKey(1) == 27) break;
}
```

## Batch Inference

One ONNX Runtime call for a whole batch instead of one call per image — the throughput
win on GPU. See [Batch Inference](../api/api.md#batch-inference) for the fallback rules.

```cpp
std::vector<cv::Mat> images = {cv::imread("a.jpg"), cv::imread("b.jpg"), cv::imread("c.jpg")};

// One result vector per input image, in input order
auto results = detector.batchDetect(images, /*conf=*/0.25f, /*iou=*/0.45f);

for (size_t i = 0; i < images.size(); ++i) {
    detector.drawDetections(images[i], results[i]);
}
```

Requires a model exported with `model.export(format="onnx", dynamic=True)`. Fixed-batch
exports fall back to a per-image loop automatically — check with
`detector.supportsBatchSize(images.size())`.

The same pattern applies to `batchSegment()`, `batchClassify()`, and `batchDetect()` on
the pose and OBB detectors.

## Loading a Model From Memory

For encrypted stores, network streams and resources embedded in the binary. See
[In-Memory Model Loading](../api/api.md#in-memory-model-loading).

```cpp
// Bytes from anywhere; readFileBytes() is just a convenience helper
std::vector<uint8_t> bytes = yolos::utils::readFileBytes("yolo11n.onnx");

// Class names as a vector, so no labels file is needed either
yolos::det::YOLODetector detector(bytes.data(), bytes.size(), {"person", "bicycle", "car"});

// ONNX Runtime copied the buffer during construction — safe to wipe it now
auto detections = detector.detect(frame);
```

## Camera Stream

```cpp
cv::VideoCapture cap(0);
cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

cv::Mat frame;
while (cap.read(frame)) {
    auto detections = detector.detect(frame);
    detector.drawDetections(frame, detections);
    cv::imshow("Live", frame);
    if (cv::waitKey(1) == 27) break;
}
```

## Performance Tips

1. **Reuse detector instances** — Create once, infer many times
2. **Use GPU when available** — 5-10x faster than CPU
3. **Adjust thresholds** — Higher confidence = fewer detections, faster NMS
4. **Match input resolution** — Use model's expected size (640x640)
5. **Batch when you have several images** — `batchDetect()` amortizes one ONNX call across the batch (needs a `dynamic=True` export)

## Error Handling

```cpp
try {
    yolos::det::YOLODetector detector("model.onnx", "labels.txt", true);
} catch (const Ort::Exception& e) {
    std::cerr << "ONNX error: " << e.what() << std::endl;
}
```

## Next Steps

- [Model Guide](../guides/models.md) — Export and optimize models
- [Development](../guides/architecture.md) — Extend the library
