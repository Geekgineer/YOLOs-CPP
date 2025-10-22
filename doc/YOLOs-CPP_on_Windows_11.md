# YOLOs-CPP

YOLOs-CPP is a C++ implementation for running YOLO models (v5, v7, v8, v9, v10, v11, v12) for object detection, instance segmentation, oriented bounding boxes (OBB), pose estimation, and quantized model support. The project uses ONNX Runtime and OpenCV for efficient inference on images, videos, and camera feeds. All setup, model export, quantization, and testing were performed by the project owner on Windows 11 (x64).

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows-Specific Instructions](#windows-specific-instructions)
- [Building the Project](#building-the-project)
- [Usage Instructions](#usage)
- [Python Scripts for Model Export](#python-scripts-for-model-export)
- [Modifications Made](#modifications-made)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Overview
This project provides a C++ framework for running YOLO models for object detection, instance segmentation, oriented bounding boxes (OBB), and pose estimation. It supports:
- Real-time inference on camera feeds (`camera_inference.cpp`).
- Processing static images (`image_inference.cpp`).
- Processing video files (`video_inference.cpp`).
- Quantized models for faster CPU inference (`yolo11n_uint8.onnx`).

All steps, from installation to testing, were executed and verified on Windows 11 (x64) using Visual Studio 2022, CMake, OpenCV 4.6.0, and ONNX Runtime.

## Prerequisites
- C++17 compatible compiler
- CMake (version 3.10 or higher)
- OpenCV (version 4.5.5 or higher)
- ONNX Runtime (version 1.16.3 or 1.19.2 recommended, with optional GPU support)
- Python Version 3.8+ with `onnxruntime` and `ultralytics` packages.
- Git for cloning the repository

## Installation

### Windows-Specific Instructions
All steps were performed by the project owner on Windows 11 (x64).

1. **Install Visual Studio 2022**:
   - Open [https://visualstudio.microsoft.com/vs/](https://visualstudio.microsoft.com/vs/).
   - Click "Free download" under "Community 2022".
   - Run the installer and select **Desktop development with C++**.
   - Ensure **MSVC v143 - VS 2022 C++ x64/x86 build tools** is checked in the right panel.
   - Click **Install** and restart your system if prompted.
   - Verify by opening Visual Studio and creating a new C++ project.

2. **Install CMake**:
   - Open [https://cmake.org/download/](https://cmake.org/download/).
   - Download the latest Windows x64 Installer.
   - Run the installer and select **Add CMake to the system PATH for all users**.
   - Verify by opening PowerShell or CMD and running:
     ```powershell
     cmake --version
     ```
     - Expected output: `cmake version 3.x.x`.

3. **Install Python 3.8+**:
   - Open [https://www.python.org/downloads/](https://www.python.org/downloads/).
   - Download the latest Python 3.8+ installer.
   - Run the installer and check **Add Python to PATH** in the first screen.
   - Install and verify by running:
     ```powershell
     python --version
     pip --version
     ```
   - Install required packages:
     ```powershell
     pip install onnx onnxruntime numpy ultralytics
     ```
   - If you see "Python was not found":
     - Open **Start**, search for "App execution aliases".
     - Turn off `python.exe` and `python3.exe`.
   - Verify Python packages:
     ```powershell
     python -c "import cv2; print(cv2.__version__)"
     ```
     - Expected output: `4.6.0` or higher.

4. **Install Git**:
   - Open [https://git-scm.com/download/win](https://git-scm.com/download/win).
   - Download and run the installer.
   - Verify by running:
     ```powershell
     git --version
     ```
     - Expected output: `git version 2.x.x`.

### Set Up DLLs
1. **Install OpenCV**:
   - Open [https://opencv.org/releases/](https://opencv.org/releases/).
   - Download OpenCV 4.6.0 for Windows.
   - Extract to `C:\opencv`.
   - Copy the following DLLs from `C:\opencv\build\x64\vc16\bin` to `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release` (after building):
     - `opencv_videoio_ffmpeg460_64.dll`
     - `opencv_videoio_msmf460_64.dll`
     - `opencv_world460.dll`
   - Add OpenCV’s `bin` directory to the system PATH:
     - Open **Start**, search for "Edit the system environment variables".
     - Click **Environment Variables**.
     - Under **System variables**, select **Path**, click **Edit**.
     - Click **New** and add:
       ```
       C:\opencv\build\x64\vc16\bin
       ```
     - Click **OK** to save all changes.
   - Restart PowerShell or CMD.
   - Verify:
     ```powershell
     python -c "import cv2; print(cv2.__version__)"
     ```
     - Expected output: `4.6.0`.

2. **Install ONNX Runtime**:
   - Open [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases).
   - Download the Windows x64 ZIP (e.g., `onnxruntime-win-x64-gpu-1.16.3.zip` for GPU support).
   - Extract to `C:\onnxruntime`.
   - Copy the following DLLs from `C:\onnxruntime\lib` to `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release` (after building):
     - `onnxruntime.dll`
     - `onnxruntime_providers_shared.dll`
     - `onnxruntime_providers_cuda.dll` (if using GPU)
     - `onnxruntime_providers_tensorrt.dll` (if using GPU)
   - Add ONNX Runtime’s `lib` directory to the system PATH:
     - Open **Environment Variables** as above.
     - Select **Path**, click **Edit**.
     - Click **New** and add:
       ```
       C:\onnxruntime\lib
       ```
     - Click **OK** to save.
   - Restart PowerShell or CMD.

#### Clone the Repository
   - Open PowerShell or CMD.
   - Run:
     ```powershell
     cd C:\Users\DELL\source\repos
     git clone https://github.com/Geekgineer/YOLOs-CPP.git
     cd YOLOs-CPP
     ```

### Export YOLO Models to ONNX
1. Navigate to the `models` directory:
   ```powershell
   cd C:\Users\DELL\source\repos\YOLOs-CPP\models
   ```
2. Run the export scripts:
   - **Object Detection**:
     ```powershell
     python export_onnx.py
     ```
     - Exports `yolo11n.pt` to `yolo11n.onnx`.
   - **Segmentation**:
     ```powershell
     python export_onnx_to_segment.py
     ```
     - Exports `yolo11n-seg.pt` to `yolo11n-seg.onnx`.
   - **OBB**:
     ```powershell
     python export_onnx_to_obb.py
     ```
     - Exports `yolo11n-obb.pt` to `yolo11n-obb.onnx`.
   - **Pose**:
     ```powershell
     python export_onnx_to_pose.py
     ```
     - Exports `yolo11n-pose.pt` to `yolo11n-pose.onnx`.
3. Verify that the ONNX files are in `C:\Users\DELL\source\repos\YOLOs-CPP\models`.

### Quantize YOLO Model
1. Navigate to the `quantized_models` directory:
   ```powershell
   cd C:\Users\DELL\source\repos\YOLOs-CPP\quantized_models
   ```
2. Run the quantization script:
   ```powershell
   python yolos_quantization.py
   ```
   - Quantizes `yolo11n.onnx` to `yolo11n_uint8.onnx` using per-channel quantization (UINT8).
3. Verify that `yolo11n_uint8.onnx` is in `C:\Users\DELL\source\repos\YOLOs-CPP\models`.

## Building the Project
1. Navigate to the project directory:
   ```powershell
   cd C:\Users\DELL\source\repos\YOLOs-CPP
   ```
2. Create and navigate to the build directory:
   ```powershell
   mkdir build
   cd build
   ```
3. Configure the project with CMake:
   ```powershell
   cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/opencv/build" -DONNXRUNTIME_DIR="C:/onnxruntime"
   ```

4. Build the project in Release mode:
   ```powershell
   cmake --build . --config Release
   ```

## Usage Instructions
All commands were tested successfully by the project owner on Windows 11 (x64). Ensure model files, label files, and input files are in the specified paths.

### Object Detection (Quantized Model)
Uses `yolo11n_uint8.onnx` for faster CPU inference.
- **Image Inference**:
  ```powershell
  cd C:\Users\DELL\source\repos\YOLOs-CPP\build
  .\Release\image_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\dog.jpg
  ```
  - Output: `output_dog.jpg` with bounding boxes.
- **Video Inference**:
  ```powershell
  .\Release\video_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\test.mp4
  ```
  - Output: `output_video.mp4` with bounding boxes.
- **Camera Inference**:
  ```powershell
  .\Release\camera_inference.exe
  ```
  - Output: PNG frames in `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release\output_frames`.

### Instance Segmentation
Uses `yolo11n-seg.onnx` with `YOLO11SegDetector` (`seg/YOLO11Seg.hpp`).
- **Image Inference**:
  ```powershell
  .\Release\image_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\dog.jpg
  ```
  - Output: `output_dog.jpg` with segmentation masks.
- **Video Inference**:
  ```powershell
  .\Release\video_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\test.mp4
  ```
  - Output: `output_video.mp4` with segmentation masks.
- **Camera Inference**:
  ```powershell
  .\Release\camera_inference.exe
  ```
  - Output: PNG frames in `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release\output_frames`.

### Oriented Bounding Boxes (OBB)
Uses `yolo11n-obb.onnx` with `YOLO11OBBDetector` (`obb/YOLO11-OBB.hpp`) and `Dota.names`.
- **Image Inference**:
  ```powershell
  .\Release\image_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\dog.jpg
  ```
  - Output: `output_dog_obb.jpg` with oriented bounding boxes.
- **Video Inference**:
  ```powershell
  .\Release\video_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\test.mp4
  ```
  - Output: `output_video_obb.mp4` with oriented bounding boxes.
- **Camera Inference**:
  ```powershell
  .\Release\camera_inference.exe
  ```
  - Output: PNG frames in `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release\output_frames_obb`.

### Pose Estimation
Uses `yolo11n-pose.onnx` with `YOLO11POSEDetector` (`pose/YOLO11-POSE.hpp`) on inputs with persons.
- **Image Inference**:
  ```powershell
  .\Release\image_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\person.jpg
  ```
  - Output: `output_pose.jpg` with keypoints and skeletons.
- **Video Inference**:
  ```powershell
  .\Release\video_inference.exe C:\Users\DELL\source\repos\YOLOs-CPP\data\test_pose.mp4
  ```
  - Output: `output_video_pose.mp4` with keypoints and skeletons.
- **Camera Inference**:
  ```powershell
  .\Release\camera_inference.exe
  ```
  - Output: PNG frames in `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release\output_frames_pose`.

## Python Scripts for Model Export
The following Python scripts were created or modified by the project owner to export and quantize YOLO models.

1. **`export_onnx.py`** (modified):
   ```python
   from ultralytics import YOLO

   # Load the YOLOv11n model
   model = YOLO("yolo11n.pt")

   # Export the model to ONNX format
   model.export(format="onnx")
   ```

2. **`export_onnx_to_segment.py`** (created):
   ```python
   from ultralytics import YOLO

   # Load the YOLOv11 segmentation model
   model = YOLO("yolo11n-seg.pt")

   # Export to ONNX format
   model.export(format="onnx", task="segment")
   ```

3. **`export_onnx_to_obb.py`** (created):
   ```python
   from ultralytics import YOLO

   # Load the YOLOv11 OBB model
   model = YOLO("yolo11n-obb.pt")

   # Export to ONNX format
   model.export(format="onnx", task="obb")
   ```

4. **`export_onnx_to_pose.py`** (created):
   ```python
   from ultralytics import YOLO

   # Load the YOLOv11 pose model
   model = YOLO("yolo11n-pose.pt")

   # Export to ONNX format
   model.export(format="onnx", task="pose")
   ```

5. **`yolos_quantization.py`** (modified):
   ```python
   from onnxruntime.quantization import quantize_dynamic, QuantType
   from pathlib import Path
   from typing import Union

   def quantize_onnx_model(onnx_model_path: Union[str, Path], quantized_model_path: Union[str, Path], per_channel: bool = False):
       """
       Quantizes an ONNX model and saves the quantized version.

       Args:
           onnx_model_path: Path to the original ONNX model file.
           quantized_model_path: Path to save the quantized model.
           per_channel: If True, quantizes weights per channel instead of per layer.
               Per-channel quantization can improve model accuracy by allowing each output channel
               to have its own scale and zero-point, which better captures the distribution of weights.
               This is especially beneficial for complex models with many channels or varying value ranges.
               Use this option when:
               - The model is complex (e.g., deep convolutional networks).
               - You observe accuracy degradation with per-layer quantization.
       """
       # Quantize the model
       quantize_dynamic(
           model_input=onnx_model_path,
           model_output=quantized_model_path,
           per_channel=per_channel,
           weight_type=QuantType.QUInt8
       )

       print("Quantization completed. Quantized model saved to:", quantized_model_path)

   if __name__ == "__main__":
       # Load the original ONNX model file path
       onnx_model_path = 'C:/Users/DELL/source/repos/YOLOs-CPP/models/yolo11n.onnx'
       # Specify the output path for the quantized model
       quantized_model_path = 'C:/Users/DELL/source/repos/YOLOs-CPP/models/yolo11n_uint8.onnx'
       # Call the quantization function
       quantize_onnx_model(onnx_model_path, quantized_model_path, per_channel=True)
   ```

6- **`CMakeLists.txt`** (modified):
   ```txt
   cmake_minimum_required(VERSION 3.0.0)
   project(yolo_ort)

   option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
   message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

   find_package(OpenCV REQUIRED)

   include_directories("include/")

   # Add executable for image inference
   add_executable(image_inference
                  src/image_inference.cpp)

   # Add executable for camera inference
   add_executable(camera_inference
                  src/camera_inference.cpp)

   # Add executable for video inference
   add_executable(video_inference
                  src/video_inference.cpp)

   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)

   # Set include directories for all executables
   target_include_directories(image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
   target_include_directories(camera_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
   target_include_directories(video_inference PRIVATE "${ONNXRUNTIME_DIR}/include")

   # Set compile features for all executables
   target_compile_features(image_inference PRIVATE cxx_std_17)
   target_compile_features(camera_inference PRIVATE cxx_std_17)
   target_compile_features(video_inference PRIVATE cxx_std_17)

   # Link libraries for all executables   
   #### Replace ${OpenCV_LIBS} with opencv_world
   target_link_libraries(image_inference opencv_world)
   target_link_libraries(camera_inference opencv_world)
   target_link_libraries(video_inference opencv_world)

   if(UNIX)
       message(STATUS "We are building on Linux!")
       # Specific Linux build commands or flags
       target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
       target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
       target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
   endif(UNIX)

   if(APPLE)
       message(STATUS "We are building on macOS!")
       # Specific macOS build commands or flags
       target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
       target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
       target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
   endif(APPLE)

   if(WIN32)
       message(STATUS "We are building on Windows!")
       # Specific Windows build commands or flags
       target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
       target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
       target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
   endif(WIN32)
```

## Modifications Made
The project owner made the following changes:
- Modified `yolos_quantization.py` to support `yolo11n.onnx` and generate `yolo11n_uint8.onnx`.
- Modified `export_onnx.py` to export `yolo11n.pt` to `yolo11n.onnx`.
- Created `export_onnx_to_segment.py`, `export_onnx_to_obb.py`, and `export_onnx_to_pose.py` to export segmentation, OBB, and pose models.
- Modified `camera_inference.cpp`, `image_inference.cpp`, and `video_inference.cpp` to support:
  - Quantized model (`yolo11n_uint8.onnx`) with `YOLO11Detector`.
  - Segmentation (`yolo11n-seg.onnx`) with `YOLO11SegDetector` and `drawSegmentations`.
  - OBB (`yolo11n-obb.onnx`) with `YOLO11OBBDetector` and `Dota.names`.
  - Pose (`yolo11n-pose.onnx`) with `YOLO11POSEDetector` and keypoints/skeletons drawing.
- Updated paths in all `.cpp` files to use absolute paths (e.g., `C:/Users/DELL/source/repos/YOLOs-CPP/models`).
- Set `isGPU = false` for CPU processing.
- Added file existence checks using `fs::exists`.
- Added detailed logging for detection time and details.
- Added saving of outputs (PNG frames for camera, images for image inference, videos for video inference).
- Tested all models and confirmed successful results with bounding boxes, masks, oriented boxes, and keypoints/skeletons.

## Troubleshooting
- **Model file does not exist**:
  - Verify that `yolo11n_uint8.onnx`, `yolo11n-seg.onnx`, `yolo11n-obb.onnx`, and `yolo11n-pose.onnx` are in `C:/Users/DELL/source/repos/YOLOs-CPP/models`.
  - Run the export scripts (`export_onnx.py`, `export_onnx_to_segment.py`, etc.) to generate missing models.
- **Labels file does not exist**:
  - Ensure `coco.names` (for detection, segmentation, pose) and `Dota.names` (for OBB) are in `C:/Users/DELL/source/repos/YOLOs-CPP/quantized_models`.
- **Could not open video/image**:
  - Check that `dog.jpg`, `person.jpg`, `test.mp4`, and `test_pose.mp4` are in `C:/Users/DELL/source/repos/YOLOs-CPP/data`.
- **No poses detected**:
  - Use inputs with persons (e.g., `person.jpg`, `test_pose.mp4`) for pose estimation.
- **DLL errors**:
  - Ensure all DLLs (`opencv_*.dll`, `onnxruntime*.dll`) are in `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release`.
  - Verify that `C:\opencv\build\x64\vc16\bin` and `C:\onnxruntime\lib` are in the system PATH.
- **CMake errors**:
  - Ensure `OpenCV_DIR` and `ONNXRUNTIME_DIR` are correctly set in the CMake command.
  - If `std::filesystem` errors occur, update `CMakeLists.txt` to use C++17 (see [Building the Project](#building-the-project)).
  - Add include directories in `CMakeLists.txt` for segmentation, OBB, and pose:
    ```cmake
    include_directories(${CMAKE_SOURCE_DIR}/seg)
    include_directories(${CMAKE_SOURCE_DIR}/obb)
    include_directories(${CMAKE_SOURCE_DIR}/pose)
    ```

## Notes
- All models were tested on CPU (`isGPU = false`) for compatibility. Enable GPU by setting `isGPU = true` if supported hardware is available (include `onnxruntime_providers_cuda.dll` and `onnxruntime_providers_tensorrt.dll`).
- Quantized models (`yolo11n_uint8.onnx`) provide faster inference with minimal accuracy loss.
- OBB requires `Dota.names` for correct class labeling.
- Pose estimation requires inputs with persons to detect keypoints and skeletons.
- Outputs are saved as:
  - Images: `output_<filename>.jpg` (e.g., `output_dog.jpg`, `output_pose.jpg`).
  - Videos: `output_video.mp4`, `output_video_obb.mp4`, `output_video_pose.mp4`.
  - Camera frames: PNGs in `output_frames`, `output_frames_obb`, or `output_frames_pose`.

### If you encounter errors related to `std::filesystem` (requiring C++17):
  - Open `CMakeLists.txt` in the project root (`C:\Users\DELL\source\repos\YOLOs-CPP`).
  - Replace like in "CMakeLists.txt (modified)":
  - Save the file.
  - Delete the `build` directory and recreate it:
    ```powershell
    cd C:\Users\DELL\source\repos\YOLOs-CPP
    Remove-Item -Recurse -Force build
    mkdir build
    cd build
    cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/opencv/build" -DONNXRUNTIME_DIR="C:/onnxruntime"
    ```   

### if there error in path 

* Copy DLLs to `build\Release`:
  - From `C:\opencv\build\x64\vc16\bin`:
    - `opencv_videoio_ffmpeg460_64.dll`
    - `opencv_videoio_msmf460_64.dll`
    - `opencv_world460.dll`
  - From `C:\onnxruntime\lib`:
    - `onnxruntime.dll`
    - `onnxruntime_providers_shared.dll`
    - `onnxruntime_providers_cuda.dll` (if using GPU)
    - `onnxruntime_providers_tensorrt.dll` (if using GPU)
  - Paste all DLLs into `C:\Users\DELL\source\repos\YOLOs-CPP\build\Release`.
