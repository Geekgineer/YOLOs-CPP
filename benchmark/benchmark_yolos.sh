#!/usr/bin/env bash
# run benchmark_yolos.sh - A comprehensive benchmarking suite for YOLOs-CPP

set -euo pipefail

# --- Configuration ---
RESULTS_DIR="../benchmark_results_$(date +%F_%H-%M-%S)"
TEST_IMAGE="../data/dogs.jpg"
TEST_VIDEO="../data/test_video.mp4" # Make sure you have a test video here
IMAGE_ITERATIONS=100

# --- Create Results Directory ---
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/benchmark_log.txt"
echo "Benchmark started: $(date)" | tee "$LOG_FILE"

# --- Compile the Benchmark Tool ---
echo "📦 Compiling C++ benchmark tool..."
# Assuming this script is run from the 'build' directory
g++ ../benchmark/bench.cpp -O3 -std=c++17 -march=native \
    -I../include \
    $(pkg-config --cflags --libs opencv4) \
    -L/opt/onnxruntime/lib -I/opt/onnxruntime/include -lonnxruntime \
    -o comprehensive_bench
echo "✅ Compile complete."

# --- CSV Headers ---
IMAGE_CSV="$RESULTS_DIR/image_benchmark.csv"
VIDEO_CSV="$RESULTS_DIR/video_benchmark.csv"

echo "model,device,precision,pre_avg_ms,infer_avg_ms,post_avg_ms,total_avg_ms,pre_med_ms,infer_med_ms,post_med_ms,total_med_ms,pre_min_ms,infer_min_ms,post_min_ms,total_min_ms,fps" > "$IMAGE_CSV"
echo "model,device,precision,total_frames,avg_ms_per_frame,avg_fps" > "$VIDEO_CSV"

# --- Helper Function ---
run_image_benchmark() {
    local model_name=$1 model_type=$2 model_path=$3 device=$4 precision=$5
    echo "▶️  Benchmarking IMAGE: ${model_name}_${device}_${precision}" | tee -a "$LOG_FILE"
    
    local device_arg=""
    if [[ "$device" == "gpu" ]]; then
        device_arg="gpu"
    fi

    local stats
    stats=$(./comprehensive_bench image "$model_type" "$model_path" "$TEST_IMAGE" "$IMAGE_ITERATIONS" "$device_arg" 2>> "$LOG_FILE") || {
        echo "❌ Benchmark failed for ${model_name} on ${device}. See log for details."
        return
    }
    
    echo "${model_name},${device},${precision},${stats}" >> "$IMAGE_CSV"
}

run_video_benchmark() {
    local model_name=$1 model_type=$2 model_path=$3 device=$4 precision=$5
    echo "▶️  Benchmarking VIDEO: ${model_name}_${device}_${precision}" | tee -a "$LOG_FILE"

    local device_arg=""
    if [[ "$device" == "gpu" ]]; then
        device_arg="gpu"
    fi

    local stats
    stats=$(./comprehensive_bench video "$model_type" "$model_path" "$TEST_VIDEO" "$device_arg" 2>> "$LOG_FILE") || {
        echo "❌ Benchmark failed for ${model_name} on ${device}. See log for details."
        return
    }

    echo "${model_name},${device},${precision},${stats}" >> "$VIDEO_CSV"
}


# --- Define Models to Test ---
# Format: "model_name model_type model_path_suffix"
MODELS_TO_TEST=(
    "yolov8n detection yolov8n.onnx"
    "yolov8s detection yolov8s.onnx"
    "yolov8n-seg segmentation yolov8n-seg.onnx"
    "yolov8n-pose pose yolov8n-pose.onnx"
    # Add other models here, e.g., yolov10, yolov11, quantized models, etc.
)

# --- Main Execution Loop ---
for model_info in "${MODELS_TO_TEST[@]}"; do
    read -r model_name model_type model_suffix <<< "$model_info"
    
    # --- FP32 Models ---
    fp32_path="../models/${model_suffix}"
    if [ -f "$fp32_path" ]; then
        run_image_benchmark "$model_name" "$model_type" "$fp32_path" "cpu" "fp32"
        run_image_benchmark "$model_name" "$model_type" "$fp32_path" "gpu" "fp32"
        echo "---" | tee -a "$LOG_FILE"
        run_video_benchmark "$model_name" "$model_type" "$fp32_path" "cpu" "fp32"
        run_video_benchmark "$model_name" "$model_type" "$fp32_path" "gpu" "fp32"
    else
        echo "⚠️  Warning: Model not found at $fp32_path. Skipping." | tee -a "$LOG_FILE"
    fi

    # --- Quantized Models (Example) ---
    uint8_path="../models/quantized/${model_suffix%.onnx}_uint8.onnx"
    if [ -f "$uint8_path" ]; then
        run_image_benchmark "${model_name}-quant" "$model_type" "$uint8_path" "cpu" "uint8"
        # GPU uint8 support can be limited, often tested on CPU
    fi
    echo "---------------------------------" | tee -a "$LOG_FILE"
done

echo "✅ All benchmarks finished!"
echo "Image results saved to: $IMAGE_CSV"
echo "Video results saved to: $VIDEO_CSV"
