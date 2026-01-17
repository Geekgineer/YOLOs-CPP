# YOLOs-CPP Examples

This directory contains task-specific examples demonstrating how to use the YOLOs-CPP library.

## Available Examples

| Example | Task | Input Type | Description |
|---------|------|------------|-------------|
| `example_image_det` | Detection | Image | Object detection on static images |
| `example_video_det` | Detection | Video | Object detection on video files |
| `example_camera_det` | Detection | Camera | Real-time object detection |
| `example_image_seg` | Segmentation | Image | Instance segmentation on images |
| `example_video_seg` | Segmentation | Video | Instance segmentation on video |
| `example_camera_seg` | Segmentation | Camera | Real-time instance segmentation |
| `example_image_pose` | Pose | Image | Human pose estimation on images |
| `example_video_pose` | Pose | Video | Pose estimation on video |
| `example_camera_pose` | Pose | Camera | Real-time pose estimation |
| `example_image_obb` | OBB | Image | Oriented bounding box detection |
| `example_video_obb` | OBB | Video | OBB detection on video |
| `example_camera_obb` | OBB | Camera | Real-time OBB detection |
| `example_image_class` | Classification | Image | Image classification |
| `example_video_class` | Classification | Video | Video classification |
| `example_camera_class` | Classification | Camera | Real-time classification |

## Building Examples

```bash
cd examples
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running Examples

### Using Shell Scripts

Each example has a corresponding shell script:

```bash
# Detection
./run_image_det.sh
./run_video_det.sh
./run_camera_det.sh

# Segmentation
./run_image_seg.sh
./run_video_seg.sh
./run_camera_seg.sh

# Pose Estimation
./run_image_pose.sh
./run_video_pose.sh
./run_camera_pose.sh

# Classification
./run_image_class.sh
./run_video_class.sh
./run_camera_class.sh

# Oriented Bounding Box
./run_image_obb.sh
./run_video_obb.sh
./run_camera_obb.sh
```

### Direct Execution

```bash
# Format: ./executable [model_path] [input_path] [labels_path]

# Detection example
./build/example_image_det ../models/yolo11n.onnx ../data/dog.jpg ../models/coco.names

# Segmentation example  
./build/example_image_seg ../models/yolo11n-seg.onnx ../data/dog.jpg ../models/coco.names

# Pose example
./build/example_image_pose ../models/yolo11n-pose.onnx ../data/person.jpg

# Classification example
./build/example_image_class ../models/yolo11n-cls.onnx ../data/dog.jpg ../models/ImageNet.names
```

## Output

Results are saved to the `outputs/` directory with timestamps:
- `outputs/det/` - Detection results
- `outputs/seg/` - Segmentation results
- `outputs/pose/` - Pose estimation results
- `outputs/obb/` - OBB detection results
- `outputs/class/` - Classification results

## Common Options

All examples support:
- **GPU acceleration**: Set `useGPU = true` in the source code
- **Custom confidence threshold**: Modify `confThreshold` parameter
- **Custom NMS threshold**: Modify `iouThreshold` parameter
- **Directory input**: Pass a directory path to process all images

## Example Code Structure

Each example follows a consistent pattern:

```cpp
#include "yolos/yolos.hpp"

int main(int argc, char* argv[]) {
    // 1. Parse arguments
    // 2. Initialize detector
    // 3. Load input (image/video/camera)
    // 4. Run inference
    // 5. Draw results
    // 6. Save/display output
    return 0;
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check the model path relative to the executable |
| CUDA not available | Ensure CUDA drivers are installed for GPU |
| Display not working | Set `DISPLAY` environment variable |
| Low FPS | Try a smaller model (e.g., yolo11n instead of yolo11x) |
