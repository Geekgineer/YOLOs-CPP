#!/bin/bash

# YOLOs-CPP RunPod Benchmark Setup Script
# This script sets up the benchmarking environment for both CPU and GPU RunPod containers

echo "=== YOLOs-CPP RunPod Benchmark Setup ==="

# Create necessary directories
mkdir -p /workspace/yolos-cpp/models
mkdir -p /workspace/yolos-cpp/data
mkdir -p /workspace/yolos-cpp/results
mkdir -p /workspace/yolos-cpp/benchmark

# Set environment variables
export RUNPOD_ENV=true
export BENCHMARK_RESULTS_DIR=/workspace/yolos-cpp/results

# Function to detect GPU availability
check_gpu_availability() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
        export GPU_AVAILABLE=true
        export ENVIRONMENT_TYPE="GPU"
    else
        echo "No GPU detected. Running in CPU-only mode."
        export GPU_AVAILABLE=false
        export ENVIRONMENT_TYPE="CPU"
    fi
}

# Function to install additional monitoring tools
install_monitoring_tools() {
    echo "Installing system monitoring tools..."
    
    # Update package list
    apt-get update -qq
    
    # Install htop for CPU monitoring
    apt-get install -y htop
    
    # Install additional monitoring utilities
    apt-get install -y sysstat  # For iostat, vmstat, etc.
    apt-get install -y procps   # For ps, top, etc.
    
    echo "Monitoring tools installed."
}

# Function to download required model files
download_models() {
    echo "Downloading YOLO model files..."
    
    cd /workspace/yolos-cpp/models
    
    # Download YOLO8 models (if not already present)
    if [ ! -f "yolo8n.onnx" ]; then
        echo "Downloading YOLO8n..."
        wget -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo8n.onnx" || echo "Failed to download yolo8n.onnx"
    fi
    
    if [ ! -f "yolo8s.onnx" ]; then
        echo "Downloading YOLO8s..."
        wget -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo8s.onnx" || echo "Failed to download yolo8s.onnx"
    fi
    
    # Download YOLO11 models (if available)
    if [ ! -f "yolo11n.onnx" ]; then
        echo "YOLO11 models should be provided separately or use existing ones"
    fi
    
    # Download label files
    if [ ! -f "coco.names" ]; then
        echo "Downloading COCO labels..."
        wget -q "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" || echo "Failed to download coco.names"
    fi
    
    echo "Model download complete."
}

# Function to download test data
download_test_data() {
    echo "Downloading test data..."
    
    cd /workspace/yolos-cpp/data
    
    # Download a test image if not present
    if [ ! -f "dog.jpg" ]; then
        echo "Downloading test image..."
        wget -q "https://github.com/pjreddie/darknet/raw/master/data/dog.jpg" || echo "Failed to download test image"
    fi
    
    # Download a test video (small sample)
    if [ ! -f "test_video.mp4" ]; then
        echo "Test video should be provided separately or use existing ones"
    fi
    
    echo "Test data download complete."
}

# Function to build the project
build_project() {
    echo "Building YOLOs-CPP project..."
    
    cd /workspace/yolos-cpp
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "Configuring for GPU build..."
        cmake .. -DONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-gpu-1.20.1 -DCMAKE_BUILD_TYPE=Release
    else
        echo "Configuring for CPU build..."
        cmake .. -DONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-1.20.1 -DCMAKE_BUILD_TYPE=Release
    fi
    
    # Build the project
    make -j$(nproc) comprehensive_bench
    
    if [ $? -eq 0 ]; then
        echo "Build successful!"
        cp comprehensive_bench ../benchmark/
    else
        echo "Build failed!"
        exit 1
    fi
}

# Function to run comprehensive benchmarks
run_benchmarks() {
    echo "Running comprehensive benchmarks..."
    
    cd /workspace/yolos-cpp/benchmark
    
    # Create timestamp for unique results
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RESULTS_FILE="$BENCHMARK_RESULTS_DIR/benchmark_results_${ENVIRONMENT_TYPE}_${TIMESTAMP}.csv"
    
    echo "Results will be saved to: $RESULTS_FILE"
    
    # Run comprehensive benchmark
    ./comprehensive_bench comprehensive > "$RESULTS_FILE" 2>&1
    
    echo "Benchmark completed. Results saved to: $RESULTS_FILE"
    
    # Also run specific benchmarks for comparison
    echo "Running specific model comparisons..."
    
    # YOLO8 vs YOLO11 comparison on available models
    for model in yolo8 yolo11; do
        if [ -f "../models/${model}n.onnx" ]; then
            echo "Benchmarking ${model}..."
            
            # CPU benchmark
            ./comprehensive_bench image $model detection "../models/${model}n.onnx" "../models/coco.names" "../data/dog.jpg" --cpu --iterations=100 >> "${RESULTS_FILE}_detailed" 2>&1
            
            # GPU benchmark (if available)
            if [ "$GPU_AVAILABLE" = true ]; then
                ./comprehensive_bench image $model detection "../models/${model}n.onnx" "../models/coco.names" "../data/dog.jpg" --gpu --iterations=100 >> "${RESULTS_FILE}_detailed" 2>&1
            fi
        fi
    done
    
    echo "Detailed benchmarks completed."
}

# Function to generate summary report
generate_report() {
    echo "Generating benchmark summary report..."
    
    REPORT_FILE="$BENCHMARK_RESULTS_DIR/benchmark_summary_${ENVIRONMENT_TYPE}_$(date +"%Y%m%d_%H%M%S").md"
    
    cat > "$REPORT_FILE" << EOF
# YOLOs-CPP Benchmark Report

## Environment Details
- **Platform**: RunPod Container
- **Environment Type**: $ENVIRONMENT_TYPE
- **Date**: $(date)
- **GPU Info**: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")

## System Specifications
- **CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
- **Memory**: $(free -h | grep "Mem:" | awk '{print $2}')
- **Disk**: $(df -h /workspace | tail -1 | awk '{print $2}')

## Benchmark Results Summary

The detailed results are available in the CSV files in this directory.

### Key Findings
- Model loading times
- Inference performance (FPS)
- Resource utilization
- Latency statistics

### Files Generated
EOF
    
    # List all result files
    ls -la "$BENCHMARK_RESULTS_DIR"/*.csv >> "$REPORT_FILE" 2>/dev/null || echo "No CSV files found" >> "$REPORT_FILE"
    
    echo "Report generated: $REPORT_FILE"
}

# Main execution flow
main() {
    echo "Starting RunPod benchmark setup..."
    
    # Check GPU availability first
    check_gpu_availability
    
    # Install monitoring tools
    install_monitoring_tools
    
    # Download required files (if needed)
    # download_models  # Comment out if models are already present
    # download_test_data  # Comment out if test data is already present
    
    # Build the project
    build_project
    
    # Run benchmarks
    run_benchmarks
    
    # Generate summary report
    generate_report
    
    echo "=== RunPod Benchmark Setup Complete ==="
    echo "Results are available in: $BENCHMARK_RESULTS_DIR"
    echo "Environment Type: $ENVIRONMENT_TYPE"
}

# Execute main function
main "$@"
