# Installation Guide

This guide covers system requirements, build options, and troubleshooting for YOLOs-CPP.

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+), Windows 10+, macOS 12+ |
| **Compiler** | GCC 9+, Clang 10+, or MSVC 2019+ |
| **CMake** | 3.16 or higher |
| **OpenCV** | 4.5 or higher |
| **C++ Standard** | C++17 |

### GPU Acceleration (Optional)

| Component | Requirement |
|-----------|-------------|
| **NVIDIA GPU** | Compute Capability 5.0+ |
| **CUDA Toolkit** | 11.0 or higher |
| **cuDNN** | 8.0+ (recommended) |

## Quick Install

### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/Geekgineer/YOLOs-CPP.git
cd YOLOs-CPP

# Build with auto-download of ONNX Runtime
./build.sh 1.20.1 0   # CPU build
./build.sh 1.20.1 1   # GPU build (requires CUDA)
```

### Windows

See [Windows Setup Guide](YOLOs-CPP_on_Windows_11.md) for detailed Windows instructions.

## Manual Build

### Step 1: Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

**macOS (Homebrew):**
```bash
brew install cmake opencv
```

### Step 2: Download ONNX Runtime

```bash
# Linux x64 CPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-1.20.1.tgz

# Linux x64 GPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
```

### Step 3: Configure and Build

```bash
mkdir build && cd build

# CPU build
cmake .. \
  -DONNXRUNTIME_DIR=../onnxruntime-linux-x64-1.20.1 \
  -DCMAKE_BUILD_TYPE=Release

# Compile
make -j$(nproc)
```

### Step 4: Verify Installation

```bash
./image_inference ../models/yolo11n.onnx ../data/dog.jpg
```

## Build Options

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `ONNXRUNTIME_DIR` | auto-detect | Path to ONNX Runtime installation |
| `BUILD_EXAMPLES` | OFF | Build task-specific examples |
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release) |

## Docker Installation

```bash
# CPU
docker build -f Dockerfile.cpu -t yolos-cpp:cpu .
docker run --rm -it yolos-cpp:cpu

# GPU (requires nvidia-docker)
docker build -t yolos-cpp:gpu .
docker run --gpus all --rm -it yolos-cpp:gpu
```

## Troubleshooting

### "OpenCV not found"

```bash
pkg-config --modversion opencv4
cmake .. -DOpenCV_DIR=/path/to/opencv/build
```

### "ONNX Runtime not found"

```bash
ls $ONNXRUNTIME_DIR/include/onnxruntime_cxx_api.h
```

### Build fails on Windows

See [Windows Setup Guide](YOLOs-CPP_on_Windows_11.md).

## Next Steps

- [Usage Guide](usage.md) — Learn the API
- [Model Guide](models.md) — Export and use models

## Windows Quick Start

### Prerequisites
- Visual Studio 2019+ with "Desktop development with C++"
- CMake 3.16+
- OpenCV 4.5+ (from opencv.org or vcpkg)

### Option 1: PowerShell Script

```powershell
# CPU build
.\build.ps1

# GPU build (requires CUDA)
.\build.ps1 -GPU

# Clean build
.\build.ps1 -Clean
```

### Option 2: Batch Script

```cmd
build.bat          # CPU build
build.bat gpu      # GPU build
```

### Option 3: Manual Build

```powershell
# Download ONNX Runtime
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-1.20.1.zip" -OutFile "ort.zip"
Expand-Archive -Path "ort.zip" -DestinationPath "."

# Build
mkdir build; cd build
cmake .. -DONNXRUNTIME_DIR="..\onnxruntime-win-x64-1.20.1"
cmake --build . --config Release

# Run
.\Release\image_inference.exe ..\models\yolo11n.onnx ..\data\dog.jpg
```

### Setting up OpenCV on Windows

**Option A: Pre-built binaries**
1. Download from https://opencv.org/releases/
2. Extract to `C:\opencv`
3. Add `C:\opencv\build\x64\vc16\bin` to PATH

**Option B: Using vcpkg**
```powershell
vcpkg install opencv4:x64-windows
cmake .. -DCMAKE_TOOLCHAIN_FILE="[vcpkg]/scripts/buildsystems/vcpkg.cmake"
```

See [Windows Setup Guide](YOLOs-CPP_on_Windows_11.md) for complete instructions.

## macOS Quick Start

```bash
# Install dependencies
brew install cmake opencv

# Download ONNX Runtime (Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz
tar -xzf onnxruntime-osx-arm64-1.20.1.tgz

# Build
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=../onnxruntime-osx-arm64-1.20.1
make -j$(sysctl -n hw.ncpu)
```

For Intel Macs, use `onnxruntime-osx-x86_64-1.20.1.tgz`.
