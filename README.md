# YOLOs-CPP

![cover](data/cover.png)


![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/language-C++-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.19.2-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen.svg)
![CMake](https://img.shields.io/badge/CMake-3.22.1-blue.svg)


## Overview
**YOLOs-CPP** provides single C++ headers with a high-performance application designed for real-time object detection, segmentation, oriented object detection (OBB), and pose estimation using various YOLO (You Only Look Once) models from [Ultralytics](https://github.com/ultralytics/ultralytics). Leveraging the power of [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [OpenCV](https://opencv.org/), this project provides seamless integration with a unified YOLOv(5,7,8,9,10,11,12) implementation for image, video, and live camera inference. Whether you're developing for research, production, or hobbyist projects, this application offers flexibility and efficiency.

## 

#### 📰 Latest Updates 📌 Pinned


* **[2025.05.15]** 🪡🪡🪡 YOLOs-CPP now supports classification.

* **[2025.04.04]** ✊✊✊ [Depths-CPP](https://github.com/Geekgineer/Depths-CPP) New Similar project for real-time metric depth estimation.

* **[2025.03.16]** 👥👥👥 YOLOs-CPP now supports pose estimation.

* **[2025.02.11]** 💯💯💯 YOLOs-CPP now supports OBB format.

* **[2025.02.19]** 🌪️🌪️🌪️ YOLOs-CPP now supports YOLOv12 for object detection.

* **[2025.01.29]** 🎯🎯🎯 YOLOs-CPP now supports YOLOv9 for object detection.

* **[2025.01.26]** 💥💥💥 YOLOs-CPP Provide now segmentation headers for YOLOv9.

* **[2025.01.26]** 🔥🔥🔥 YOLOs-CPP Provide now segmentation headers for YOLOv8 and YOLOv11 also quantized models.

* **[2024.10.23]** 🚀🚀🚀 YOLOs-CPP Project launch with support for detection headers.

#### 🛠️ Open Tasks for Contribution

* [ ] Benchmarking
* [ ] Write documentation


*Video example of object detection output with segmentation masks, bounding boxes and labels. [Click on image!]*

<table>
  <tr>
    <td>
      <a href="https://www.youtube.com/watch?v=Ax5vaYJ-mVQ">
        <img src="data/SIG_experience_center_seg_processed.gif" alt="Watch the Demo Video" width="400" height="225"/>
      </a>
    </td>
    <td>
      <a href="https://www.youtube.com/watch?v=Ax5vaYJ-mVQ">
        <img src="data/SIG_experience_center_seg_processed-2.gif" alt="Watch the Demo Video" width="400" height="225"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <img src="data/final_test_compressed_output.gif" alt="Demo GIF" width="400" height="225"/>
    </td>
    <td>
          <img src="data/dance_output.gif" alt="Demo GIF" width="400" height="225"/>
    </td>
  </tr>
</table>


### Integration in your c++ projects

## Detection Example

```cpp

// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "det/YOLO11.hpp" // Ensure YOLO11.hpp or other version is in your include path

int main()
{
    // Configuration parameters
    const std::string labelsPath = "../models/coco.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n.onnx";     // Path to YOLO11 model
    const std::string imagePath  = "../data/dogs.jpg";           // Path to input image
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the YOLO11 detector
    YOLO11Detector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform object detection to get bboxs
    std::vector<Detection> detections = detector.detect(image);

    // Draw bounding boxes on the image
    detector.drawBoundingBoxMask(image, detections);

    // Display the annotated image
    cv::imshow("YOLO11 Detections", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}


```

## Segmentation Example

```cpp

// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Include the YOLOv11 Segmentation header
#include "seg/YOLO11Seg.hpp"

int main()
{
    // Configuration parameters
    const std::string labelsPath = "../models/coco.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n-seg.onnx";     // Path to YOLO11 model
    const std::string imagePath  = "../data/dogs.jpg";           // Path to input image
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the YOLO11 segmentor
    YOLOv11SegDetector segmentor(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform object segmentation to get segmentation masks and bboxs
    std::vector<Segmentation> results = detector.segment(img, 0.2f, 0.45f);

    // Draw bounding boxes on the image
    segmentor.drawSegmentations(image, results);          // Masks only
    // segmentor.drawSegmentationsAndBoxes(image, results); // Masks and Detections

    // Display the annotated image
    cv::imshow("YOLO11 Segmentation and Detections", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}

```

## Oriented Detection Example

```cpp

// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Include the YOLOv11 OBB header
#include "obb/YOLO11-OBB.hpp" 

int main()
{
    // Configuration parameters
    const std::string labelsPath = "../models/Dota.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n-obb.onnx";     // Path to YOLO11-OBB model
    const std::string imagePath  = "../data/frame_37.jpg";           // Path to input image
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the YOLO11 Detector
    YOLO11OBBDetector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform object detection to get oriented bounding boxes
    std::vector<Detection> results = detector.detect(image);

    // Draw bounding boxes on the image
    detector.drawBoundingBox(image, results);  

    // Display the annotated image
    cv::imshow("YOLO11 Oriented object Detections", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}

```

## Pose Estimation Example

```cpp
// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Include the YOLO11-POSE Pose header
#include "pose/YOLO11-POSE.hpp" 

int main()
{
    // Configuration parameters
    const std::string labelsPath = "../models/coco.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n-pose.onnx"; // Path to YOLO11 Pose model
    const std::string imagePath  = "../data/person.jpg";         // Path to input image
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the YOLO11 pose detector
    YOLO11POSEDetector poseDetector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform pose estimation
    std::vector<PoseDetection> poses = poseDetector.detect(image);

    // Draw keypoints and skeletons on the image
    poseDetector.drawBoundingBox(image, poses);

    // Display the annotated image
    cv::imshow("YOLO11 Pose Estimation", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}

```



> **Note:** For more usage, check the source files: [camera_inference.cpp](src/camera_inference.cpp), [image_inference.cpp](src/image_inference.cpp), [video_inference.cpp](src/video_inference.cpp).

## Features


- **Multiple YOLO Models**: Supports YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11 and YOLOv12 with standard and quantized ONNX models for flexibility in use cases.
  
- **ONNX Runtime Integration**: Leverages ONNX Runtime for optimized inference on both CPU and GPU, ensuring high performance.
  - **Dynamic Shapes Handling**: Adapts automatically to varying input sizes for improved versatility.
  - **Graph Optimization**: Enhances performance using model optimization with `ORT_ENABLE_ALL`.
  - **Execution Providers**: Configures sessions for CPU or GPU (e.g., `CUDAExecutionProvider` for GPU support).
  - **Input/Output Shape Management**: Manages dynamic input tensor shapes per model specifications.
  - **Optimized Memory Allocation**: Utilizes `Ort::MemoryInfo` for efficient memory management during tensor creation.
  - **Batch Processing**: Supports processing multiple images, currently focused on single-image input.
  - **Output Tensor Extraction**: Extracts output tensors dynamically for flexible result handling.

- **OpenCV Integration**: Uses OpenCV for image processing and rendering bounding boxes and labels (note: `cv::dnn` modules are not used).

- **Real-Time Inference**: Capable of processing images, videos, and live camera feeds instantly.

- **Efficient Detection Handling**: Employs Non-Maximum Suppression (NMS) for effective processing (note: some models are NMS free e.g. YOLO10).

- **Efficient Oriented Detection Handling**: Employs Rotated Non-Maximum Suppression (NMS) for effective Oriented Bounding Box processing.

- **Efficient Pose Estimation Handling**: Optimizes keypoint-based filtering and Non-Maximum Suppression (NMS) for accurate and reliable pose estimation.

- **Cross-Platform Support**: Fully compatible with Linux, macOS, and Windows environments.

- **Easy-to-Use Scripts**: Includes shell scripts for straightforward building and running of different inference modes.


## Requirements

Before building the project, ensure that the following dependencies are installed on your system:

- **C++ Compiler**: Compatible with C++14 standard (e.g., `g++`, `clang++`, or MSVC).
- **CMake**: Version 3.0.0 or higher.
- **OpenCV**: Version 4.5.5 or higher.
- **ONNX Runtime**: Tested with version 1.16.3 and 1.19.2, backward compatibility [Installed and linked automatically during the build].
- **Python** (optional): For running the quantization script (`yolos_quantization.py`).

## Installation

### Clone Repository

First, clone the repository to your local machine:

```bash 
git clone https://github.com/Geekgineer/YOLOs-CPP
cd YOLOs-CPP
```

### Configure

1. make sure you have opencv c++ installed
2. set the ONNX Runtime version you need e.g. ONNXRUNTIME_VERSION="1.16.3" in [build.sh](build.sh) to download ONNX Runtime headers also set GPU.
3. Ensure that the you commented and uncomment the yolo version you need in the script you will run, select the data or the camera source in the code:
   
    [camera_inference.cpp](src/camera_inference.cpp)

    [image_inference.cpp](src/image_inference.cpp)

    [video_inference.cpp](src/video_inference.cpp)

4. Optional: control the debugging and timing using [Config.hpp](tools/Config.hpp)



### Build the Project

Execute the build script to compile the project using CMake:

```bash
./build.sh
```

This script will download onnxruntime headers, create a build directory, configure the project, and compile the source code. Upon successful completion, the executable files (camera_inference, image_inference, video_inference) will be available in the build directory.

### Usage

After building the project, you can perform object detection on images, videos, or live camera feeds using the provided shell scripts.

#### Run Image Inference

To perform object detection on a single image:
Ensure you have an input image (e.g., dog.jpg) in the correct directory before running the script:

```bash
./run_image.sh 
```
This command will process the specified image using a YOLO model and display the output with detected bounding boxes and labels.


#### Run Video Inference

To perform object detection on a video file:
Make sure you have a video file available before running:

```bash
./run_video.sh 
```
The script will process the provided video using the YOLO model and save the output with detected objects.

#### Run Camera Inference

To perform real-time object detection using a usb cam:

```bash
./run_camera.sh 
```

This command will activate your usb and display the video feed with real-time object detection.

### YOLOs Models
While ONNX provides a cross-platform format for model compatibility, exporting the model directly for the target device or optimizing it for specific hardware can significantly improve performance. To achieve the best inference speed and resource efficiency, it's generally recommended to tailor the export process to the hardware on which the model will run.

The project includes several pre-trained and pre-exported standard YOLO models, located in `models` and `quantized_models` [Cloud Drive](https://mega.nz/folder/TvgXVRQJ#6M0IZdMOvKlKY9-dx7Uu7Q) directories. However, it’s not recommended to use these directly. Instead, you should always export your PyTorch models using the [export_onnx.py](./models/export_onnx.py) script.


| Model Type       | Model Name                |
|------------------|---------------------------|
| **Standard Models**    | yolo5-n6.onnx              |
|                  | yolo7-tiny.onnx            |
|                  | yolo8n.onnx                |
|                  | yolo8n-seg.onnx                |
|                  | yolo8n-obb.onnx                |
|                  | yolov8n-pose.onnx               |
|                  | yolov9s.onnx               |
|                  | yolo10n.onnx               |
|                  | yolo11n.onnx               |
|                  | yolo11n-seg.onnx               |
|                  | yolo11n-obb.onnx               |
|                  | yolo11n-pose.onnx              |
|                  | yolo12n.onnx               |
| **Quantized Models**   | yolo5-n6_uint8.onnx         |
|                  | yolo7-tiny-uint8.onnx      |
|                  | yolo8n_uint8.onnx          |
|                  | yolo8n-seg_uint8.onnx          |
|                  | yolo10n_uint8.onnx         |
|                  | yolo11n_uint8.onnx         |
|                  | yolo11n-seg_uint8.onnx         |

You can also use your custom YOLO version with your own custom classes!


**Class Names:**
- coco.names: Contains the list of class labels used by the models.
- Dota.names: Contains the list of class labels used by the OBB models.

### Quantization

The quantized_models directory includes quantized versions of the YOLO models optimized for lower precision inference. Additionally, the `quantized_models/yolos_quantization.py` script can be used to perform custom quantization on your custom YOLOs models.

> Note: Quantized models offer reduced model size and potentially faster inference with a slight trade-off in accuracy.


### Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**: Click the "Fork" button at the top-right corner of this repository to create a personal copy.
2. **Clone Your Fork**:
    ```bash
    git clone https://github.com/Geekgineer/YOLOs-CPP
    cd YOLOs-CPP
    ```
3. **Create a New Branch**:
    ```bash
    git checkout -b feature/YourFeatureName
    ```
4. **Make Your Changes**: Implement your feature or bug fix.
5. **Commit Your Changes**:
    ```bash
    git commit -m "Add feature: YourFeatureName"
    ```
6. **Push to Your Fork**:
    ```bash
    git push origin feature/YourFeatureName
    ```
7. **Create a Pull Request**: Navigate to the original repository and click "New Pull Request" to submit your changes for review.

### License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.

### Acknowledgment

- [https://github.com/itsnine/yolov5-onnxruntime](https://github.com/itsnine/yolov5-onnxruntime)
- [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [https://github.com/Li-99/yolov8_onnxruntime](https://github.com/Li-99/yolov8_onnxruntime)
- [https://github.com/K4HVH/YOLOv8-ONNXRuntime-CPP](https://github.com/K4HVH/YOLOv8-ONNXRuntime-CPP)
- [https://github.com/iamstarlee/YOLOv8-ONNXRuntime-CPP](https://github.com/iamstarlee/YOLOv8-ONNXRuntime-CPP)
