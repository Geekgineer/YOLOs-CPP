#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
# Hardcode ONNX Runtime GPU version and archive
ONNXRUNTIME_VERSION="1.20.1"
ONNXRUNTIME_PLATFORM="linux"
ONNXRUNTIME_ARCH="x64"
ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
ONNXRUNTIME_FILE="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-gpu-${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_ARCHIVE_EXTENSION}"
ONNXRUNTIME_DIR="${CURRENT_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-gpu-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"

# RTX 4090 specific build environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=1
export CUDA_AUTO_BOOST=1

# Download and extract ONNX Runtime GPU only if not already present
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "Downloading ONNX Runtime GPU from $ONNXRUNTIME_URL ..."
    curl -L -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"
    echo "Extracting ONNX Runtime GPU ..."
    tar -xzf "${ONNXRUNTIME_FILE}" -C "$CURRENT_DIR"
    rm -f "${ONNXRUNTIME_FILE}"
else
    echo "ONNX Runtime GPU already exists. Skipping download."
fi

# Build the project with RTX 4090 optimizations
BUILD_DIR="${CURRENT_DIR}/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake with RTX 4090 optimizations..."

# RTX 4090 specific compiler flags
RTX4090_CXX_FLAGS="-O3 -march=native -mtune=native -msse4.2 -mavx2 -mfma -DWITH_CUDA -DCUDA_ARCH_BIN=\"8.9\" -DENABLE_FAST_MATH"
RTX4090_CUDA_FLAGS="-O3 -use_fast_math -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -Xptxas -O3"

# Configure CMake with specific optimizations
cmake .. \
    -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="$RTX4090_CXX_FLAGS" \
    -DCUDA_NVCC_FLAGS="$RTX4090_CUDA_FLAGS" \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DENABLE_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

echo "Building YOLOs-CPP..."
echo "Using $(nproc) CPU cores for parallel compilation..."
cmake --build . --config Release -- -j$(nproc)

echo "✅ Build completed successfully!"
