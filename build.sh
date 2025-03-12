#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)

# Default values
ONNXRUNTIME_VERSION="${1:-1.20.1}"
ONNXRUNTIME_GPU="${2:-0}"

# Function to display usage
usage() {
    echo "Usage: $0 [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]"
    echo
    echo "This script downloads ONNX Runtime for the current platform and architecture and builds YOLOs-CPP."
    echo
    echo "Arguments:"
    echo "  ONNXRUNTIME_VERSION   Version of ONNX Runtime to download (default: 1.20.1)."
    echo "  ONNXRUNTIME_GPU       Whether to use GPU support (0 for CPU, 1 for GPU, default: 0)."
    echo
    echo "Examples:"
    echo "  $0 1.20.1 0          # Downloads ONNX Runtime v1.20.1 for CPU."
    echo "  $0 1.16.3 1          # Downloads ONNX Runtime v1.16.3 for GPU."
    echo
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Detect platform and architecture
platform=$(uname -s)
architecture=$(uname -m)

case "$platform" in
Darwin*)
    ONNXRUNTIME_PLATFORM="osx"
    ;;
Linux*) 
    ONNXRUNTIME_PLATFORM="linux"
    ;;
MINGW*) 
    ONNXRUNTIME_PLATFORM="win"
    ;;
*)
    echo "Unsupported platform: $platform"
    exit 1
    ;;
esac

# Determine ONNX Runtime architecture
case "$architecture" in
aarch64|arm64)
    ONNXRUNTIME_ARCH="aarch64"
    ;;
x86_64)
    ONNXRUNTIME_ARCH="x64"
    ;;
arm*)
    ONNXRUNTIME_ARCH="arm"
    ;;
i*86)
    ONNXRUNTIME_ARCH="x86"
    ;;
*)
    echo "Unsupported architecture: $architecture"
    exit 1
    ;;
esac

# Set the correct ONNX Runtime download filename
ONNXRUNTIME_FILE="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"

if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-gpu"
fi

ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.tgz"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"
ONNXRUNTIME_DIR="${CURRENT_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}"

# Function to download and extract ONNX Runtime
download_onnxruntime() {
    echo "Downloading ONNX Runtime from $ONNXRUNTIME_URL ..."
    
    if ! curl -L -C - -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"; then
        echo "Error: Failed to download ONNX Runtime."
        exit 1
    fi

    echo "Extracting ONNX Runtime ..."
    if ! tar -zxvf "${ONNXRUNTIME_FILE}" -C "$CURRENT_DIR"; then
        echo "Error: Failed to extract ONNX Runtime."
        exit 1
    fi

    rm -f "${ONNXRUNTIME_FILE}"
}

# Function to build the project
build_project() {
    local build_type="${1:-Release}"
    local build_dir="${CURRENT_DIR}/build"

    # Ensure the build directory exists
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "Configuring CMake with build type: $build_type ..."
    cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo "Building project incrementally ..."
    cmake --build . -- -j$(nproc)  # Parallel build using available CPU cores
}



# Main script execution
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    download_onnxruntime
else
    echo "ONNX Runtime already exists. Skipping download."
fi

build_project "Release"

echo "Build completed successfully."
