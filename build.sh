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
    ONNXRUNTIME_GPU=0
    ;;
Linux*) ONNXRUNTIME_PLATFORM="linux" ;;
MINGW*) ONNXRUNTIME_PLATFORM="win" ;;
*)
    echo "Unsupported platform: $platform"
    exit 1
    ;;
esac

# Determine ONNX Runtime architecture
if [[ "$architecture" == "aarch64" || "$architecture" == "arm64" ]]; then
    ONNXRUNTIME_GPU=0
    if [[ "$ONNXRUNTIME_PLATFORM" == "linux" ]]; then
        ONNXRUNTIME_ARCH="aarch64"
    else
        ONNXRUNTIME_ARCH="arm64"
    fi
elif [[ "$architecture" == "x86_64" ]]; then
    if [[ "$ONNXRUNTIME_PLATFORM" == "win" ]]; then
        ONNXRUNTIME_ARCH="x64"
    else
        ONNXRUNTIME_ARCH="x86_64"
    fi
elif [[ "$architecture" == arm* ]]; then
    ONNXRUNTIME_ARCH="arm"
elif [[ "$architecture" == i*86 ]]; then
    ONNXRUNTIME_ARCH="x86"
else
    echo "Unsupported architecture: $architecture"
    exit 1
fi

# Determine ONNX Runtime path
if [ "$ONNXRUNTIME_GPU" -eq 1 ]; then
    ONNXRUNTIME_PATH="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-gpu-${ONNXRUNTIME_VERSION}"
else
    ONNXRUNTIME_PATH="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}"
fi

ONNXRUNTIME_DIR="${CURRENT_DIR}/${ONNXRUNTIME_PATH}"

# Function to download and extract ONNX Runtime
download_onnxruntime() {
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_PATH}.tgz"
    echo "Downloading ONNX Runtime from $url ..."
    if ! curl -L -O -C - "$url"; then
        echo "Failed to download ONNX Runtime."
        exit 1
    fi

    echo "Extracting ONNX Runtime ..."
    if ! tar -zxvf "${ONNXRUNTIME_PATH}.tgz"; then
        echo "Failed to extract ONNX Runtime."
        exit 1
    fi

    rm -f "${ONNXRUNTIME_PATH}.tgz"
}

# Function to build the project
build_project() {
    local build_type="${1:-Release}"
    local build_dir="${CURRENT_DIR}/build"

    if [ -d "$build_dir" ]; then
        echo "Removing previous build directory ..."
        rm -rf "$build_dir"
    fi

    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "Configuring CMake with build type: $build_type ..."
    cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo "Building project ..."
    cmake --build .
}

# Main script execution
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    download_onnxruntime
fi

build_project "Release"

echo "Build completed successfully."
