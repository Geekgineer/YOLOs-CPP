# Installation Guide for YOLOs-CPP

## Prerequisites
Ensure the following are installed on your system:

- **C++ Compiler**: Compatible with C++14 (e.g., `g++`, `clang++`, MSVC)
- **CMake**: Version 3.0.0 or higher
- **OpenCV**: Version 4.5.5 or higher
- **ONNX Runtime**: Installed automatically via build script (versions 1.16.3 and 1.19.2 supported)
- **Python (Optional)**: For quantization script

## Clone the Repository
```bash
git clone https://github.com/Geekgineer/YOLOs-CPP
cd YOLOs-CPP
```

## Configuration Steps

1. **Ensure OpenCV is Installed**
   - Required for image processing and rendering

2. **Set ONNX Runtime Version**
   - Open `build.sh`
   - Set `ONNXRUNTIME_VERSION="1.16.3"` or desired version
   - Toggle GPU usage by modifying build flags

3. **Select YOLO Version and Data Source**
   - Edit one of the following files to choose your YOLO version and data input:
     - `src/camera_inference.cpp`
     - `src/image_inference.cpp`
     - `src/video_inference.cpp`

4. **Optional Debug Settings**
   - Edit `tools/Config.hpp` to enable or disable debugging/timing logs

## Build the Project
Run the build script to configure and compile the project:
```bash
./build.sh
```
This script will:
- Download ONNX Runtime headers
- Configure the CMake project
- Build all inference executables

### Output
The following binaries will be available in the `build/` directory:
- `image_inference`
- `video_inference`
- `camera_inference`

## Troubleshooting
- Ensure OpenCV and CMake are in your system path
- Check permissions for `build.sh`
- Confirm you have network access to fetch ONNX headers

For detailed usage, see `docs/USAGE.md`.

