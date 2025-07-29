# Developer Guide: Integrating YOLOs-CPP

## 💼 Header-Only Integration
Each detection mode is encapsulated in its own header-only class:

### Detection
```cpp
#include "det/YOLO11.hpp"
YOLO11Detector detector(modelPath, labelsPath, isGPU);
```

### Segmentation
```cpp
#include "seg/YOLO11Seg.hpp"
YOLOv11SegDetector segmentor(modelPath, labelsPath, isGPU);
```

### Oriented Detection (OBB)
```cpp
#include "obb/YOLO11-OBB.hpp"
YOLO11OBBDetector detector(modelPath, labelsPath, isGPU);
```

### Pose Estimation
```cpp
#include "pose/YOLO11-POSE.hpp"
YOLO11POSEDetector poseDetector(modelPath, labelsPath, isGPU);
```

---

## 🧠 ONNX Runtime Highlights
- Uses `Ort::Session` for model loading
- Execution providers:
  - `CPUExecutionProvider`
  - `CUDAExecutionProvider`
- Dynamic shape handling
- Optimizations enabled via `ORT_ENABLE_ALL`
- Efficient memory using `Ort::MemoryInfo`

## 🎨 OpenCV Usage
- Used for input image decoding, rendering results, and visualization.
- Drawing utilities include:
  - `drawBoundingBox`
  - `drawBoundingBoxMask`
  - `drawSegmentations`
  - `drawSegmentationsAndBoxes`
  - `drawKeypoints`

## 🛠 Tools and Debugging
- Modify `tools/Config.hpp` to toggle DEBUG or TIME_LOG features
- Add timing checks using OpenCV or std::chrono

## 📦 Batch Support (Planned)
- Current support is single image inference
- Roadmap includes batch processing capabilities

For model usage, see `docs/USAGE.md` and `docs/MODELS.md`.

