#!/usr/bin/env bash
set -euo pipefail

mode="$1"  # image | video | camera
model="$2"
labels="$3"
target="$4"  # path or camera index for "camera"
iters="${5:-50}"  # only used for image mode
use_gpu="${6:-}"   # if “gpu” specified

# compile
g++ -DTIMING_MODE benchmark.cpp -std=gnu++17 -O3 -march=native \
    -I../include \
    -I../onnxruntime-linux-x64-1.20.1/include \
    -I../onnxruntime-linux-x64-1.20.1/include/onnxruntime/core/session \
    $(pkg-config --cflags --libs opencv4) \
    -L../onnxruntime-linux-x64-1.20.1/lib -lonnxruntime \
    -o comprehensive_bench
echo "Compiled 👌"

run_cmd="./comprehensive_bench $mode \"$model\" \"$labels\" \"$target\""
if [[ "$mode" == "image" ]]; then
  run_cmd="$run_cmd $iters"
fi
if [[ "$use_gpu" == "gpu" ]]; then
  run_cmd="$run_cmd gpu"
fi

echo "▶️ Running: $run_cmd"
eval $run_cmd
