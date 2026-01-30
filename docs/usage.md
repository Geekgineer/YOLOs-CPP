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

## Error Handling

```cpp
try {
    yolos::det::YOLODetector detector("model.onnx", "labels.txt", true);
} catch (const Ort::Exception& e) {
    std::cerr << "ONNX error: " << e.what() << std::endl;
}
```

## Next Steps

- [Model Guide](models.md) — Export and optimize models
- [Development](development.md) — Extend the library
