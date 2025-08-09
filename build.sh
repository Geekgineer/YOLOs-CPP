#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.20.1}"

# Detect platform
case "$(uname -s)" in
  Linux*)  ORT_PLATFORM="linux";  EXT="tgz" ;;
  Darwin*) ORT_PLATFORM="osx";    EXT="tgz" ;;
  CYGWIN*|MINGW*|MSYS*) ORT_PLATFORM="win"; EXT="zip" ;;
  *) echo "Unsupported platform: $(uname -s)"; exit 1 ;;
esac

# Detect arch
case "$(uname -m)" in
  x86_64|amd64) ORT_ARCH="x64" ;;
  aarch64|arm64) ORT_ARCH="arm64" ;;
  *) echo "Unsupported arch: $(uname -m)"; exit 1 ;;
esac

# GPU only for Linux here (Windows/macOS handled via native scripts)
ORT_GPU=0
if [[ "$ORT_PLATFORM" == "linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  ORT_GPU=1
  echo "GPU support detected, using ONNX Runtime GPU"
else
  echo "Using ONNX Runtime CPU"
fi

# Compose filenames/paths
ORT_BASENAME="onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}"
[[ "$ORT_GPU" -eq 1 ]] && ORT_BASENAME="${ORT_BASENAME}-gpu"

ORT_FILE="${ORT_BASENAME}-${ONNXRUNTIME_VERSION}.${EXT}"
ORT_DIR="${CURRENT_DIR}/${ORT_BASENAME}-${ONNXRUNTIME_VERSION}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ORT_FILE}"

# Download/extract if needed
if [[ ! -d "$ORT_DIR" ]]; then
  echo "Downloading ONNX Runtime: $ORT_URL"
  if command -v curl >/dev/null 2>&1; then curl -L -o "$ORT_FILE" "$ORT_URL"
  elif command -v wget >/dev/null 2>&1; then wget -O "$ORT_FILE" "$ORT_URL"
  else echo "Need curl or wget"; exit 1; fi

  echo "Extracting..."
  if [[ "$EXT" == "tgz" ]]; then tar -xzf "$ORT_FILE" -C "$CURRENT_DIR"
  else unzip -q "$ORT_FILE" -d "$CURRENT_DIR"; fi
  rm -f "$ORT_FILE"
else
  echo "ONNX Runtime already exists"
fi

# Runtime lib path (helps run-after-build without exporting manually)
case "$ORT_PLATFORM" in
  linux) export LD_LIBRARY_PATH="${ORT_DIR}/lib:${LD_LIBRARY_PATH:-}";;
  osx)   export DYLD_LIBRARY_PATH="${ORT_DIR}/lib:${DYLD_LIBRARY_PATH:-}";;
  win)   export PATH="${ORT_DIR}/lib;${PATH}";; # for Git Bash
esac

# CUDA env (Linux GPU)
if [[ "$ORT_GPU" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
fi

# CMake configure/build
BUILD_DIR="${CURRENT_DIR}/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake…"
CMAKE_ARGS=(
  "-D" "ONNXRUNTIME_DIR=${ORT_DIR}"
  "-D" "CMAKE_BUILD_TYPE=Release"
  "-D" "CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
)


# Compiler flags
if [[ "$ORT_GPU" -eq 1 ]]; then
  # x86 SIMD only on x86_64
  if [[ "$(uname -m)" =~ ^(x86_64|amd64)$ ]]; then
    CXX_FLAGS="-O3 -march=native -mtune=native -msse4.2 -mavx2 -mfma"
  else
    CXX_FLAGS="-O3 -march=native -mtune=native"
  fi
  CMAKE_ARGS+=("-D" "CMAKE_CXX_FLAGS_RELEASE=${CXX_FLAGS}")

  CUDA_ARCH="${CUDA_ARCH:-}"
  if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')" || true
  fi
  [[ -z "$CUDA_ARCH" ]] && CUDA_ARCH="86"

  CMAKE_ARGS+=(
    "-D" "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
  )
else
  CMAKE_ARGS+=("-D" "CMAKE_CXX_FLAGS_RELEASE=-O3 -march=native -mtune=native")
fi

cmake .. "${GENERATOR[@]}" "${CMAKE_ARGS[@]}"

CORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --build . --config Release -- -j"${CORES}"

echo "Build completed."
