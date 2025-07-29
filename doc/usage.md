# Usage Guide for YOLOs-CPP

## Running Inference
After building, you can run inference using these scripts:

### Image Inference
```bash
./run_image.sh
```
- Ensure the image file (e.g. `dogs.jpg`) is placed in the `data/` folder.
- The output will display bounding boxes, masks, or keypoints depending on the selected model.

### Video Inference
```bash
./run_video.sh
```
- Place your video file in the data path defined in `src/video_inference.cpp`.
- Output will be shown in a display window and/or saved to file.

### Camera Inference
```bash
./run_camera.sh
```
- Uses USB webcam (default index 0)
- Displays real-time detection output

## Choosing Models
- Select the correct header and model file path inside your C++ source file.
  - Example: `#include "det/YOLO11.hpp"`
- Supported model types:
  - Detection: `YOLO11Detector`
  - Segmentation: `YOLOv11SegDetector`
  - Oriented Detection: `YOLO11OBBDetector`
  - Pose Estimation: `YOLO11POSEDetector`

## Input and Output
- Input: Images/videos loaded via OpenCV
- Output: Modified image with visual overlays (bounding boxes, masks, keypoints)

## Example Paths
```cpp
const std::string labelsPath = "../models/coco.names";
const std::string modelPath  = "../models/yolo11n.onnx";
const std::string imagePath  = "../data/dogs.jpg";
```

## Notes
- Toggle CPU/GPU in constructor using `bool isGPU = true;`
- Models and class label files should match (e.g., `coco.names`, `Dota.names`)

## Debugging
To enable debug prints and performance logs:
```cpp
// Edit tools/Config.hpp
#define DEBUG true
#define TIMING true
```

See `docs/MODELS.md` for more information about available models.

