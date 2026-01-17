#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$CURRENT_DIR/.." && pwd)

# Default values
ONNXRUNTIME_VERSION="${1:-1.20.1}"
ONNXRUNTIME_GPU="${2:-0}"
MODELS="${3:-yolo11n,yolov8n,yolo26n}"
EVAL_DATASET="${4:-}"

# Function to display usage
usage() {
    echo "Usage: $0 [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU] [MODELS] [EVAL_DATASET]"
    echo
    echo "This script downloads ONNX Runtime, builds the unified benchmark tool, and runs comprehensive benchmarks."
    echo
    echo "Arguments:"
    echo "  ONNXRUNTIME_VERSION   Version of ONNX Runtime to download (default: 1.20.1)."
    echo "  ONNXRUNTIME_GPU       Whether to use GPU support (0 for CPU, 1 for GPU, default: 0)."
    echo "  MODELS                Comma-separated list of models to test (default: yolo11n,yolov8n)."
    echo "  EVAL_DATASET          Path to evaluation dataset folder (optional)."
    echo
    echo "Examples:"
    echo "  $0 1.20.1 0                    # CPU build and benchmark"
    echo "  $0 1.20.1 1                    # GPU build and benchmark"
    echo "  $0 1.20.1 0 yolo11n            # Test single model"
    echo "  $0 1.20.1 0 yolo11n,yolov8n ../val2017  # With evaluation dataset"
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
    ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
    ;;
Linux*) 
    ONNXRUNTIME_PLATFORM="linux"
    ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
    ;;
MINGW*) 
    ONNXRUNTIME_PLATFORM="win"
    ONNXRUNTIME_ARCHIVE_EXTENSION="zip"
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
ONNXRUNTIME_DIR="${PROJECT_ROOT}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"

if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
    ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-gpu"
    ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-gpu"
fi

ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_ARCHIVE_EXTENSION}"
ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"

# Function to download and extract ONNX Runtime
download_onnxruntime() {
    echo "Downloading ONNX Runtime from $ONNXRUNTIME_URL ..."
    
    cd "$PROJECT_ROOT"
    
    if ! curl -L -C - -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"; then
        echo "Error: Failed to download ONNX Runtime."
        exit 1
    fi

    echo "Extracting ONNX Runtime ..."
    if [[ "${ONNXRUNTIME_ARCHIVE_EXTENSION}" = "tgz" ]]; then
        if ! tar -zxvf "${ONNXRUNTIME_FILE}" -C "$PROJECT_ROOT"; then
            echo "Error: Failed to extract ONNX Runtime."
            exit 1
        fi
    elif [[ "${ONNXRUNTIME_ARCHIVE_EXTENSION}" = "zip" ]]; then
        if ! unzip "${ONNXRUNTIME_FILE}" -d "$PROJECT_ROOT"; then
            echo "Error: Failed to extract ONNX Runtime."
            exit 1
        fi
    else
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

    echo "Building unified benchmark tool ..."
    cmake --build . -- -j$(nproc)  # Parallel build using available CPU cores
    
    echo "Build completed successfully."
}

# Function to run benchmarks
run_benchmarks() {
    local benchmark_exe="${CURRENT_DIR}/build/yolo_unified_benchmark"
    
    if [[ ! -f "$benchmark_exe" ]]; then
        echo "Error: Benchmark executable not found: $benchmark_exe"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    echo ""
    echo "=========================================="
    echo "Running Comprehensive Benchmarks"
    echo "=========================================="
    echo ""
    
    # Run comprehensive benchmark
    "$benchmark_exe" comprehensive
    
    # If evaluation dataset is provided, run accuracy evaluation
    if [[ -n "$EVAL_DATASET" && -d "$EVAL_DATASET" ]]; then
        echo ""
        echo "=========================================="
        echo "Running Accuracy Evaluation"
        echo "=========================================="
        echo ""
        
        # Check for ground truth labels
        GT_LABELS_DIR="${EVAL_DATASET}/../labels_val2017"
        if [[ ! -d "$GT_LABELS_DIR" ]]; then
            GT_LABELS_DIR="${EVAL_DATASET}/labels"
        fi
        
        if [[ -d "$GT_LABELS_DIR" ]]; then
            IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
            for model in "${MODEL_ARRAY[@]}"; do
                model_path="models/${model}.onnx"
                if [[ ! -f "$model_path" ]]; then
                    echo "Warning: Model not found: $model_path, skipping..."
                    continue
                fi
                
                echo "Evaluating model: $model"
                "$benchmark_exe" evaluate yolo11 detection "$model_path" models/coco.names "$EVAL_DATASET" "$GT_LABELS_DIR" --gpu
                echo ""
            done
        else
            echo "Warning: Ground truth labels directory not found. Skipping accuracy evaluation."
            echo "Expected locations: ${EVAL_DATASET}/../labels_val2017 or ${EVAL_DATASET}/labels"
        fi
    else
        echo ""
        echo "No evaluation dataset provided. Skipping accuracy evaluation."
        echo "To run accuracy evaluation, provide dataset path as 4th argument."
    fi
    
    echo ""
    echo "=========================================="
    echo "Benchmarking Complete!"
    echo "=========================================="
    echo "Results saved in: results/"
    echo ""
}

# Main script execution
echo "=========================================="
echo "YOLO Unified Benchmark - Auto Build & Run"
echo "=========================================="
echo ""

# Check dependencies
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed. Please install CMake 3.16 or higher."
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed. Please install curl."
    exit 1
fi

# Download ONNX Runtime if needed
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    download_onnxruntime
else
    echo "ONNX Runtime already exists. Skipping download."
fi

# Build the project
build_project "Release"

# Run benchmarks
run_benchmarks

echo "All done!"

