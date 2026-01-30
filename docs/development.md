# Development Guide

Architecture overview, extending YOLOs-CPP, and debugging.

## Architecture

```
include/yolos/
├── core/                    # Shared utilities
│   ├── types.hpp           # Detection, Segmentation types
│   ├── preprocessing.hpp   # Letterbox, normalization
│   ├── nms.hpp             # Non-maximum suppression
│   ├── drawing.hpp         # Visualization
│   ├── version.hpp         # YOLO version detection
│   ├── utils.hpp           # Helper functions
│   └── session_base.hpp    # ONNX session wrapper
├── tasks/                   # Task implementations
│   ├── detection.hpp       # YOLODetector
│   ├── segmentation.hpp    # YOLOSegDetector
│   ├── pose.hpp            # YOLOPoseDetector
│   ├── obb.hpp             # YOLOOBBDetector
│   └── classification.hpp  # YOLOClassifier
└── yolos.hpp               # Master include
```

## Core Components

### Preprocessing (`preprocessing.hpp`)

```cpp
// Letterbox with padding
cv::Mat blob = yolos::preprocessing::letterBoxToBlob(
    image,
    cv::Size(640, 640),
    scalePad  // Returns scale and padding info
);
```

### NMS (`nms.hpp`)

```cpp
// Class-aware batched NMS
std::vector<int> indices;
yolos::nms::NMSBoxesFBatched(
    boxes, scores, classIds,
    confThreshold, iouThreshold,
    indices
);
```

### Drawing (`drawing.hpp`)

```cpp
yolos::drawing::drawBoundingBox(image, box, label, color);
yolos::drawing::drawMask(image, mask, color, alpha);
yolos::drawing::drawKeypoints(image, keypoints);
```

## Adding a New YOLO Version

### Step 1: Update Version Enum

```cpp
// include/yolos/core/version.hpp
enum class YOLOVersion {
    V5, V6, V7, V8, V9, V10, V11, V12, V26,
    VNew  // Add your version
};
```

### Step 2: Implement Postprocessing

```cpp
// include/yolos/tasks/detection.hpp
void postprocessVNew(/* params */) {
    // Parse model output
    // Apply NMS
    // Return detections
}
```

### Step 3: Update Factory

```cpp
switch (version) {
    case YOLOVersion::VNew:
        return postprocessVNew(...);
    // ...
}
```

### Step 4: Add Tests

```cpp
// tests/detection/compare_results.cpp
// Add model to test suite
```

## ONNX Runtime Integration

### Session Management

```cpp
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOs-CPP");
Ort::SessionOptions options;

// CPU optimization
options.SetIntraOpNumThreads(4);
options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

// GPU (CUDA)
OrtCUDAProviderOptions cuda_options{};
options.AppendExecutionProvider_CUDA(cuda_options);

Ort::Session session(env, "model.onnx", options);
```

### Memory Efficiency

```cpp
// Pre-allocate buffers
std::vector<float> inputBuffer(3 * 640 * 640);

// Create tensor from existing memory
Ort::Value::CreateTensor<float>(
    memoryInfo,
    inputBuffer.data(),
    inputBuffer.size(),
    inputShape.data(),
    inputShape.size()
);
```

## Debugging

### Enable Verbose Output

```cpp
Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "Debug");
```

### Profile Inference

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
auto detections = detector.detect(frame);
auto end = std::chrono::high_resolution_clock::now();

auto ms = std::chrono::duration<double, std::milli>(end - start).count();
std::cout << "Inference: " << ms << " ms" << std::endl;
```

### Validate Against Python

```bash
# Run comparison tests
cd tests
./test_detection.sh
```

## Code Style

- **C++17** standard
- **snake_case** for variables and functions
- **PascalCase** for classes and types
- **UPPER_CASE** for constants
- Use `const` and `[[nodiscard]]` where appropriate

## Building Tests

```bash
cd tests
./build_test.sh 0  # Detection
./build_test.sh 1  # Classification
./build_test.sh 2  # Segmentation
./build_test.sh 3  # Pose
./build_test.sh 4  # OBB
```

## Benchmarking

```bash
cd benchmarks
./auto_bench.sh 1.20.1 0 yolo11n,yolov8n
```

## Next Steps

- [Contributing](contributing.md) — Submit changes
- [Model Guide](models.md) — Model compatibility
