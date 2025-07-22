# #!/bin/bash

# set -euo pipefail

# CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)

# # Default values
# ONNXRUNTIME_VERSION="${1:-1.20.1}"
# ONNXRUNTIME_GPU="${2:-0}"

# # Function to display usage
# usage() {
#     echo "Usage: $0 [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]"
#     echo
#     echo "This script downloads ONNX Runtime for the current platform and architecture and builds YOLOs-CPP."
#     echo
#     echo "Arguments:"
#     echo "  ONNXRUNTIME_VERSION   Version of ONNX Runtime to download (default: 1.20.1)."
#     echo "  ONNXRUNTIME_GPU       Whether to use GPU support (0 for CPU, 1 for GPU, default: 0)."
#     echo
#     echo "Examples:"
#     echo "  $0 1.20.1 0          # Downloads ONNX Runtime v1.20.1 for CPU."
#     echo "  $0 1.16.3 1          # Downloads ONNX Runtime v1.16.3 for GPU."
#     echo
#     exit 1
# }

# # Show usage if help is requested
# if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
#     usage
# fi

# # Detect platform and architecture
# platform=$(uname -s)
# architecture=$(uname -m)

# case "$platform" in
# Darwin*)
#     ONNXRUNTIME_PLATFORM="osx"
# 	ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
#     ;;
# Linux*) 
#     ONNXRUNTIME_PLATFORM="linux"
# 	ONNXRUNTIME_ARCHIVE_EXTENSION="tgz"
#     ;;
# MINGW*) 
#     ONNXRUNTIME_PLATFORM="win"
# 	ONNXRUNTIME_ARCHIVE_EXTENSION="zip"
#     ;;
# *)
#     echo "Unsupported platform: $platform"
#     exit 1
#     ;;
# esac

# # Determine ONNX Runtime architecture
# case "$architecture" in
# aarch64|arm64)
#     ONNXRUNTIME_ARCH="aarch64"
#     ;;
# x86_64)
#     ONNXRUNTIME_ARCH="x64"
#     ;;
# arm*)
#     ONNXRUNTIME_ARCH="arm"
#     ;;
# i*86)
#     ONNXRUNTIME_ARCH="x86"
#     ;;
# *)
#     echo "Unsupported architecture: $architecture"
#     exit 1
#     ;;
# esac

# # Set the correct ONNX Runtime download filename
# ONNXRUNTIME_FILE="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
# ONNXRUNTIME_DIR="${CURRENT_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"

# if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
#     ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-gpu"
#     ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-gpu"
# fi

# ONNXRUNTIME_FILE="${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_ARCHIVE_EXTENSION}"
# ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}-${ONNXRUNTIME_VERSION}"
# ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"

# # Function to download and extract ONNX Runtime
# download_onnxruntime() {
#     echo "Downloading ONNX Runtime from $ONNXRUNTIME_URL ..."
    
#     if ! curl -L -C - -o "${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL"; then
#         echo "Error: Failed to download ONNX Runtime."
#         exit 1
#     fi

#     echo "Extracting ONNX Runtime ..."
# 	if [[ "${ONNXRUNTIME_ARCHIVE_EXTENSION}" = "tgz" ]]; then
# 		if ! tar -zxvf "${ONNXRUNTIME_FILE}" -C "$CURRENT_DIR"; then
# 			echo "Error: Failed to extract ONNX Runtime."
# 			exit 1
# 		fi
# 	elif [[ "${ONNXRUNTIME_ARCHIVE_EXTENSION}" = "zip" ]]; then
# 		if ! unzip "${ONNXRUNTIME_FILE}" -d "$CURRENT_DIR"; then
# 			echo "Error: Failed to extract ONNX Runtime."
# 			exit 1
# 		fi
# 	else
# 		echo "Error: Failed to extract ONNX Runtime."
# 		exit 1
# 	fi

#     rm -f "${ONNXRUNTIME_FILE}"
# }

# # Function to build the project
# build_project() {
#     local build_type="${1:-Release}"
#     local build_dir="${CURRENT_DIR}/build"

#     # Ensure the build directory exists
#     mkdir -p "$build_dir"
#     cd "$build_dir"

#     echo "Configuring CMake with build type: $build_type ..."
#     cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

#     echo "Building project incrementally ..."
#     cmake --build . -- -j$(nproc)  # Parallel build using available CPU cores
# }



# # Main script execution
# if [ ! -d "$ONNXRUNTIME_DIR" ]; then
#     download_onnxruntime
# else
#     echo "ONNX Runtime already exists. Skipping download."
# fi

# build_project "Release"

# echo "Build completed successfully."
#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)

# RTX 4090 Optimized Build Configuration
echo "=== RTX 4090 Optimized Build Script ==="

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

# Configure CMake with RTX 4090 specific optimizations
cmake .. \
    -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="$RTX4090_CXX_FLAGS" \
    -DCUDA_NVCC_FLAGS="$RTX4090_CUDA_FLAGS" \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DENABLE_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

echo "Building YOLOs-CPP with RTX 4090 optimizations..."
echo "Using $(nproc) CPU cores for parallel compilation..."
cmake --build . --config Release -- -j$(nproc)

echo "✅ Build completed successfully with RTX 4090 optimizations!"
echo "RTX 4090 Optimizations Applied:"
echo "  - CUDA Compute Capability: 8.9 (Ada Lovelace)"
echo "  - Fast Math: Enabled"
echo "  - TF32: Enabled via NVIDIA_TF32_OVERRIDE"
echo "  - Compiler optimizations: -O3 -march=native -mtune=native"
echo "  - CUDA optimizations: -use_fast_math -arch=sm_89"
