# YOLOs-CPP Examples

This directory contains ready-to-run C++ examples demonstrating all core YOLOs-CPP tasks. Each example is designed to be simple, well-documented, and easy to understand.

## 📁 Structure

```
examples/
├── example_image_*.cpp    # Image processing examples
├── example_video_*.cpp    # Video processing examples  
├── example_camera_*.cpp   # Real-time camera examples
├── run_*.sh               # Shell scripts to run examples
├── utils.hpp              # Utility functions (timestamps, saving, etc.)
├── CMakeLists.txt         # Build configuration
└── README.md              # This file
```

## 🎯 Available Examples

### 📷 Image Examples
| Example | Description | Run Script |
|---------|-------------|------------|
| `example_image_det.cpp` | Standard object detection | `./run_image_det.sh` |
| `example_image_seg.cpp` | Instance segmentation | `./run_image_seg.sh` |
| `example_image_obb.cpp` | Oriented bounding boxes | `./run_image_obb.sh` |
| `example_image_class.cpp` | Image classification | `./run_image_class.sh` |
| `example_image_pose.cpp` | Pose estimation | `./run_image_pose.sh` |

### 🎥 Video Examples
| Example | Description | Run Script |
|---------|-------------|------------|
| `example_video_det.cpp` | Video object detection | `./run_video_det.sh` |
| `example_video_seg.cpp` | Video segmentation | `./run_video_seg.sh` |
| `example_video_obb.cpp` | Video OBB detection | `./run_video_obb.sh` |
| `example_video_class.cpp` | Video classification | `./run_video_class.sh` |
| `example_video_pose.cpp` | Video pose estimation | `./run_video_pose.sh` |

### 📹 Camera Examples
| Example | Description | Run Script |
|---------|-------------|------------|
| `example_camera_det.cpp` | Real-time detection | `./run_camera_det.sh` |
| `example_camera_seg.cpp` | Real-time segmentation | `./run_camera_seg.sh` |
| `example_camera_obb.cpp` | Real-time OBB detection | `./run_camera_obb.sh` |
| `example_camera_class.cpp` | Real-time classification | `./run_camera_class.sh` |
| `example_camera_pose.cpp` | Real-time pose estimation | `./run_camera_pose.sh` |

## 🚀 Quick Start

### 1. Build Examples

From the project root:

```bash
# Build using the main project's build system
cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
make
```

Or build examples separately:

```bash
cd examples
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
make -j$(nproc)
```

### 2. Run Examples

Each example has a corresponding shell script for easy execution:

```bash
# Run with default settings
./run_image_det.sh

# Run with custom model and input
./run_image_det.sh ../models/yolo11n.onnx ../data/dog.jpg

# Run with all custom arguments
./run_image_det.sh <model_path> <input_path> <labels_path>
```

## 📝 Usage Patterns

### Image Examples

Process single images or entire directories:

```bash
# Single image
./run_image_det.sh ../models/yolo11n.onnx ../data/dog.jpg

# Directory of images
./run_image_det.sh ../models/yolo11n.onnx ../data/images/

# Custom labels
./run_image_det.sh ../models/yolo11n.onnx ../data/dog.jpg ../models/coco.names
```

### Video Examples

Process video files:

```bash
# Default video
./run_video_det.sh

# Custom video
./run_video_det.sh ../models/yolo11n.onnx ../data/traffic.mp4

# Press 'q' to stop processing
```

### Camera Examples

Real-time processing from webcam:

```bash
# Default camera (camera 0)
./run_camera_det.sh

# Specific camera
./run_camera_det.sh ../models/yolo11n.onnx ../models/coco.names 0

# Press 'q' to quit, 's' to save snapshot
```

## 📤 Output

All examples save results with timestamps to the `outputs/` directory:

```
outputs/
├── det/      # Detection results
├── seg/      # Segmentation results
├── obb/      # OBB detection results
├── class/    # Classification results
└── pose/     # Pose estimation results
```

Output filename format: `{original_name}_{timestamp}_result.{ext}`

Example: `dog_20251016_143052_result.jpg`

## 🔧 Configuration

### Default Settings

All examples use **CPU by default** for inference. To enable GPU:

1. Edit the example source file
2. Change: `bool useGPU = false;` to `bool useGPU = true;`
3. Rebuild

### Supported Models

| Task | Default Model | Alternative Models |
|------|---------------|-------------------|
| Detection | `yolo11n.onnx` | `yolo8n.onnx`, `yolo10n.onnx` |
| Segmentation | `yolo11n-seg.onnx` | `yolo8n-seg.onnx` |
| OBB | `yolo11n-obb.onnx` | `yolo8n-obb.onnx` |
| Classification | `yolo11n-cls.onnx` | `yolo8n-cls.onnx` |
| Pose | `yolo11n-pose.onnx` | `yolo8n-pose.onnx` |

### Labels Files

- **COCO dataset**: `models/coco.names` (80 classes)
- **DOTA dataset** (OBB): `models/Dota.names` (15 classes)
- **ImageNet** (Classification): `models/imagenet_classes.txt` (1000 classes)

## 🎨 Features

Each example includes:

✅ **Command-line arguments** - Flexible model and input paths  
✅ **Directory support** - Process multiple files automatically  
✅ **Timestamped outputs** - Automatic result saving with timestamps  
✅ **Performance metrics** - FPS, inference time, and object counts  
✅ **Visual feedback** - Real-time display with annotations  
✅ **Error handling** - Graceful error messages and recovery  

## 🛠️ Customization

### Adding New Examples

1. Create `example_<type>_<task>.cpp` in `examples/`
2. Include appropriate YOLO header from `../include/`
3. Use `utils.hpp` for timestamps and saving
4. Add executable to `CMakeLists.txt`
5. Create corresponding `run_<type>_<task>.sh` script

### Modifying Examples

All examples follow a consistent structure:

```cpp
#include <opencv2/opencv.hpp>
#include "<task>/YOLO-<TASK>.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    // 1. Parse arguments
    // 2. Initialize detector
    // 3. Load input
    // 4. Run inference
    // 5. Save and display results
    return 0;
}
```

## 📊 Performance Tips

1. **GPU Acceleration**: Enable GPU for 5-10x speedup
2. **Model Selection**: Use nano (`n`) models for speed, large (`l`) for accuracy
3. **Input Resolution**: Lower resolution = faster processing
4. **Batch Processing**: Process multiple images in one session

## 🐛 Troubleshooting

### Build Errors

```bash
# Ensure ONNXRUNTIME_DIR is set correctly
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime

# Check OpenCV installation
pkg-config --modversion opencv4
```

### Runtime Errors

- **Model not found**: Check model path (default: `../models/`)
- **Input not found**: Check input path or use absolute paths
- **Camera not opening**: Try different camera ID (0, 1, 2...)
- **GPU not available**: Ensure CUDA-enabled ONNX Runtime build

### Output Issues

- **No output saved**: Check `outputs/` directory exists and has write permissions
- **Poor quality**: Try higher resolution or larger model variant

## 📚 Additional Resources

- [Main README](../README.md) - Project overview
- [YOLO Headers](../include/) - Header-only implementations
- [Models Directory](../models/) - Pre-trained models
- [Build Documentation](../doc/) - Detailed build instructions

## 🤝 Contributing

Found a bug or want to add an example? Please:

1. Fork the repository
2. Create your feature branch
3. Test thoroughly
4. Submit a pull request

## 📄 License

These examples are part of YOLOs-CPP and are licensed under the MIT License.

---

**Happy Coding! 🚀**

For questions or issues, please open an issue on GitHub.



