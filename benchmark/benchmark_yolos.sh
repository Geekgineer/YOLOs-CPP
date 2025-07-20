#!/usr/bin/env bash
set -euo pipefail

# 📁 Configuration
SRC="bench.cpp"
OUTPUT="comprehensive_bench"
MODEL="../models/yolo11n.onnx"
LABELS="../models/coco.names"
IMAGE="../data/dog.jpg"          # set your test image path
VIDEO="../data/dogs.mp4"          # set your test video path
ITER=50                   # default iterations for image mode
USE_GPU="gpu"             # set to "" if GPU not needed

# 🛠 Compile
echo "🔧 Compiling $SRC ..."
g++ -DTIMING_MODE "$SRC" -std=gnu++17 -O3 \
  -I../include \
  -I../onnxruntime-linux-x64-1.20.1/include \
  -I../onnxruntime-linux-x64-1.20.1/include/onnxruntime/core/session \
  $(pkg-config --cflags --libs opencv4) \
  -L../onnxruntime-linux-x64-1.20.1/lib -lonnxruntime \
  -o "$OUTPUT"
echo "✅ Compiled: $OUTPUT"

# 📦 Helper to run and log
run() {
  local mode="$1"
  shift
  echo "▶️ Running ${mode} benchmark..."
  echo "Command: ./$OUTPUT $mode $* $USE_GPU"
  mkdir -p logs
  ./"$OUTPUT" "$mode" "$@" $USE_GPU | tee "logs/${mode}_$(date +%F_%H%M%S).log"
  echo ""
}

# 🧪 Image Mode
run image "$MODEL" "$LABELS" "$IMAGE" "$ITER"

# 🎬 Video Mode
run video "$MODEL" "$LABELS" "$VIDEO"

# 📷 Live Camera Mode (default camera)
run video "$MODEL" "$LABELS" "0"

echo "✅ All benchmarks complete. Check the logs/ folder for details."
