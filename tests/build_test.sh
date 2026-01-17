#!/bin/bash
# ============================================================================
# YOLOs-CPP Test Build Script
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Default values
TEST_TASK="${1:-0}"           # 0=detection, 1=classification, 2=segmentation, 3=pose, 4=obb, 5=all
ONNXRUNTIME_VERSION="${2:-1.20.1}"
ONNXRUNTIME_GPU="${3:-0}"     # 0=CPU, 1=GPU

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Usage
# ============================================================================
usage() {
    echo "Usage: $0 [TEST_TASK] [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]"
    echo ""
    echo "Arguments:"
    echo "  TEST_TASK           Task to test:"
    echo "                        0 = Detection"
    echo "                        1 = Classification"
    echo "                        2 = Segmentation"
    echo "                        3 = Pose"
    echo "                        4 = OBB"
    echo "                        5 = All tasks (default)"
    echo "  ONNXRUNTIME_VERSION Version of ONNX Runtime (default: 1.20.1)"
    echo "  ONNXRUNTIME_GPU     0 = CPU, 1 = GPU (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run detection tests with CPU"
    echo "  $0 5 1.20.1 0       # Run all tests with CPU"
    echo "  $0 0 1.20.1 1       # Run detection with GPU"
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# ============================================================================
# Platform Detection
# ============================================================================
detect_platform() {
    local platform=$(uname -s)
    local arch=$(uname -m)

    case "$platform" in
        Darwin*) ONNXRUNTIME_PLATFORM="osx"; ARCHIVE_EXT="tgz" ;;
        Linux*)  ONNXRUNTIME_PLATFORM="linux"; ARCHIVE_EXT="tgz" ;;
        MINGW*)  ONNXRUNTIME_PLATFORM="win"; ARCHIVE_EXT="zip" ;;
        *) echo -e "${RED}Unsupported platform: $platform${NC}"; exit 1 ;;
    esac

    case "$arch" in
        aarch64|arm64) ONNXRUNTIME_ARCH="aarch64" ;;
        x86_64)        ONNXRUNTIME_ARCH="x64" ;;
        arm*)          ONNXRUNTIME_ARCH="arm" ;;
        i*86)          ONNXRUNTIME_ARCH="x86" ;;
        *) echo -e "${RED}Unsupported architecture: $arch${NC}"; exit 1 ;;
    esac

    # Build ONNX Runtime paths
    ONNXRUNTIME_FILE="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
    ONNXRUNTIME_DIR="${SCRIPT_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"

    if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
        ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-gpu"
        ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-gpu"
    fi

    ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.${ARCHIVE_EXT}"
    ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-${ONNXRUNTIME_VERSION}"
    ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"
}

# ============================================================================
# Download ONNX Runtime
# ============================================================================
download_onnxruntime() {
    if [[ -d "$ONNXRUNTIME_DIR" ]]; then
        echo -e "${GREEN}ONNX Runtime already exists: $ONNXRUNTIME_DIR${NC}"
        return
    fi

    echo -e "${YELLOW}Downloading ONNX Runtime from $ONNXRUNTIME_URL...${NC}"
    
    if ! curl -L -C - -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"; then
        echo -e "${RED}Failed to download ONNX Runtime${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Extracting ONNX Runtime...${NC}"
    if [[ "$ARCHIVE_EXT" == "tgz" ]]; then
        tar -xzf "${ONNXRUNTIME_FILE}" -C "$SCRIPT_DIR"
    else
        unzip "${ONNXRUNTIME_FILE}" -d "$SCRIPT_DIR"
    fi

    rm -f "${ONNXRUNTIME_FILE}"
    echo -e "${GREEN}ONNX Runtime extracted to: $ONNXRUNTIME_DIR${NC}"
}

# ============================================================================
# Build Project
# ============================================================================
build_project() {
    local build_dir="${SCRIPT_DIR}/build"

    mkdir -p "$build_dir"
    cd "$build_dir"

    echo -e "${YELLOW}Configuring CMake...${NC}"
    cmake .. \
        -DONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
        -DtestTask="${TEST_TASK}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3"

    echo -e "${YELLOW}Building...${NC}"
    cmake --build . -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    echo -e "${GREEN}Build completed successfully!${NC}"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================"
echo "  YOLOs-CPP Test Build"
echo "============================================"
echo "  Task:          $TEST_TASK"
echo "  ONNX Runtime:  $ONNXRUNTIME_VERSION"
echo "  GPU:           $ONNXRUNTIME_GPU"
echo "============================================"

detect_platform
download_onnxruntime
build_project

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo "Run tests with: cd build && ctest --output-on-failure"
