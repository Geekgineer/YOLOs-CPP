# YOLOs-CPP

![cover](data/cover.png)


![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/language-C++-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.19.2-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen.svg)
![CMake](https://img.shields.io/badge/CMake-3.22.1-blue.svg)


## Overview

**YOLOs-CPP** is a high-performance C++ library for real-time object detection, segmentation, oriented object detection (OBB), and pose estimation using multiple YOLO model versions. It integrates ONNX Runtime and OpenCV to support fast, flexible inference across a variety of input types (image, video, camera).

## Features

- **Multiple YOLO Models**: Support for YOLOv5 to YOLOv12
- **Detection Types**: Standard detection, segmentation, OBB, and pose estimation
- **Backends**: ONNX Runtime for GPU/CPU acceleration
- **Real-Time**: Optimized for real-time performance
- **Cross-Platform**: Linux, Windows, macOS
- **Easy Integration**: Modular headers and examples for C++ projects

## 🔄 Recent Updates

- **[2025.05.15]**: Classification support added
- **[2025.03.16]**: Pose estimation support
- **[2025.02.11]**: OBB support
- **[2025.01.29]**: YOLOv9+ support
- **[2024.10.23]**: Initial release

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/Geekgineer/YOLOs-CPP
cd YOLOs-CPP
```

### Build

```bash
./build.sh
```

### Run Inference

- Image: `./run_image.sh`
- Video: `./run_video.sh`
- Camera: `./run_camera.sh`

## Supported Models

| Type         | Examples                   |
| ------------ | -------------------------- |
| Standard     | yolo11n.onnx, yolo12n.onnx |
| Segmentation | yolo11n-seg.onnx           |
| OBB          | yolo11n-obb.onnx           |
| Pose         | yolo11n-pose.onnx          |
| Quantized    | yolo11n\_uint8.onnx        |

Custom ONNX export recommended via `models/export_onnx.py`.

## 🎥 Demo Gallery

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


> For full installation, usage, contribution, and model details, see the `docs/` folder.

---

### License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.

### Acknowledgments

See `docs/ACKNOWLEDGMENTS.md` for external contributions and references.

