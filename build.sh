#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)

# Auto-detect platform and architecture
ONNXRUNTIME_VERSION="1.20.1"

# Detect platform
case "$(uname -s)" in
    Linux*)
        ONNXRUNTIME_PLATFORM="linux"
        ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
        ;;
    Darwin*)
        ONNXRUNTIME_PLATFORM="osx"
        ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        ONNXRUNTIME_PLATFORM="win"
        ONNXRUNTIME_ARCHIVE_EXTENSION="zip"
        ;;
    *)
        echo "❌ Unsupported platform: $(uname -s)"
        echo "Supported platforms: Linux, macOS, Windows"
        exit 1
        ;;
esac

# Detect architecture
case "$(uname -m)" in
    x86_64|amd64)
        ONNXRUNTIME_ARCH="x64"
        ;;
    aarch64|arm64)
        ONNXRUNTIME_ARCH="arm64"
        ;;
    *)
        echo "❌ Unsupported architecture: $(uname -m)"
        echo "Supported architectures: x86_64, aarch64"
        exit 1
        ;;
esac

# GPU support detection (Linux only for now)
ONNXRUNTIME_GPU=0
if [[ "$ONNXRUNTIME_PLATFORM" == "linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    ONNXRUNTIME_GPU=1
    echo "🔥 GPU support detected, using ONNX Runtime GPU"
else
    echo "💻 Using ONNX Runtime CPU"
fi

# Construct download URL and paths
ONNXRUNTIME_FILE="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
ONNXRUNTIME_DIR="${CURRENT_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"

if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-gpu"
    ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-gpu"
fi

ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_ARCHIVE_EXTENSION}"
ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"


# Set CUDA environment variables (Linux GPU only)
if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export NVIDIA_TF32_OVERRIDE=1
    export CUDA_AUTO_BOOST=1
fi

# Download and extract ONNX Runtime only if not already present
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "📥 Downloading ONNX Runtime from $ONNXRUNTIME_URL ..."
    
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"
    else
        echo "❌ Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    echo "📦 Extracting ONNX Runtime ..."
    if [[ "$ONNXRUNTIME_ARCHIVE_EXTENSION" == "tgz" ]]; then
        tar -xzf "${ONNXRUNTIME_FILE}" -C "$CURRENT_DIR"
    elif [[ "$ONNXRUNTIME_ARCHIVE_EXTENSION" == "zip" ]]; then
        unzip -q "${ONNXRUNTIME_FILE}" -d "$CURRENT_DIR"
    fi
    rm -f "${ONNXRUNTIME_FILE}"
else
    echo "✅ ONNX Runtime already exists. Skipping download."
fi

BUILD_DIR="${CURRENT_DIR}/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "🔧 Configuring CMake with platform optimizations..."

# Platform-specific optimizations
CMAKE_ARGS=()
CMAKE_ARGS+=("-D ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")

if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    # GPU-specific optimizations (Linux)
    CXX_FLAGS="-O3 -march=native -mtune=native -msse4.2 -mavx2 -mfma -DWITH_CUDA -DCUDA_ARCH_BIN=\"8.9\" -DENABLE_FAST_MATH"
    CUDA_FLAGS="-O3 -use_fast_math -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -Xptxas -O3"
    
    CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS_RELEASE=$CXX_FLAGS")
    CMAKE_ARGS+=("-DCUDA_NVCC_FLAGS=$CUDA_FLAGS")
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=89")
    CMAKE_ARGS+=("-DENABLE_CUDA=ON")
    CMAKE_ARGS+=("-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda")
else
    # CPU-only optimizations
    CXX_FLAGS="-O3 -march=native -mtune=native"
    CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS_RELEASE=$CXX_FLAGS")
fi

CMAKE_ARGS+=("-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON")

# Configure CMake
cmake .. "${CMAKE_ARGS[@]}"

echo "🔨 Building YOLOs-CPP..."
echo "Using $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) CPU cores for parallel compilation..."
cmake --build . --config Release -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "✅ Build completed successfully!"
echo ""
echo "🎯 Available executables:"
echo "  • ./build/yolo_performance_analyzer - Advanced benchmarking tool"
echo "  • ./build/yolo_benchmark_suite - Quick comparison tool"
echo "  • ./build/image_inference - Single image inference"
echo "  • ./build/video_inference - Video processing"
echo "  • ./build/camera_inference - Real-time camera inference"
