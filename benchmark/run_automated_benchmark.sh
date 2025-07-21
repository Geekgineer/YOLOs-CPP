#!/bin/bash

# YOLOs-CPP Automated Benchmark Script
# Runs comprehensive benchmarks and generates analysis reports

set -e  # Exit on any error

echo "=== YOLOs-CPP Automated Benchmark System ==="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
MODELS_DIR="$PROJECT_ROOT/models"
DATA_DIR="$PROJECT_ROOT/data"

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$DATA_DIR"

# Function to check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    
    # Check if benchmark executable exists
    if [ ! -f "$SCRIPT_DIR/comprehensive_bench" ]; then
        echo "Error: comprehensive_bench not found. Please build the project first."
        echo "Run: cd $PROJECT_ROOT/build && make comprehensive_bench && cp comprehensive_bench ../benchmark/"
        exit 1
    fi
    
    # Check for required model files
    local required_models=("yolo11n.onnx" "yolov8n.onnx")
    for model in "${required_models[@]}"; do
        if [ ! -f "$MODELS_DIR/$model" ]; then
            echo "Warning: Required model $model not found in $MODELS_DIR"
        fi
    done
    
    # Check for test data
    if [ ! -f "$DATA_DIR/dog.jpg" ]; then
        echo "Warning: Test image dog.jpg not found in $DATA_DIR"
    fi
    
    echo "Dependency check completed."
}

# Function to detect environment
detect_environment() {
    echo "Detecting environment..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        export HAS_GPU=true
        export GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 | tr ' ' '_')
    else
        echo "No GPU detected. Running CPU-only benchmarks."
        export HAS_GPU=false
        export GPU_NAME="CPU_Only"
    fi
    
    # Get CPU info
    export CPU_NAME=$(lscpu | grep "Model name" | cut -d: -f2 | xargs | tr ' ' '_')
    export CPU_CORES=$(nproc)
    export TOTAL_MEMORY=$(free -m | awk 'NR==2{printf "%s", $2}')
    
    echo "Environment: $CPU_NAME, $CPU_CORES cores, ${TOTAL_MEMORY}MB RAM"
}

# Function to run image benchmarks
run_image_benchmarks() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_file="$RESULTS_DIR/image_benchmark_${timestamp}.csv"
    
    echo "Running image benchmarks..."
    echo "Results will be saved to: $results_file"
    
    # Available models to test
    local models=()
    [ -f "$MODELS_DIR/yolov8n.onnx" ] && models+=("yolo8")
    [ -f "$MODELS_DIR/yolo11n.onnx" ] && models+=("yolo11")
    
    if [ ${#models[@]} -eq 0 ]; then
        echo "No supported models found. Skipping image benchmarks."
        return
    fi
    
    # Create header
    echo "timestamp,model_type,task_type,environment,device,threads,precision,load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,latency_avg_ms,latency_min_ms,latency_max_ms,map_score,frame_count,cpu_name,gpu_name" > "$results_file"
    
    local test_iterations=(50 100 200)
    local thread_counts=(1 4 8)
    
    for model in "${models[@]}"; do
        # Handle different naming conventions
        if [ "$model" = "yolo8" ]; then
            local model_file="$MODELS_DIR/yolov8n.onnx"
        else
            local model_file="$MODELS_DIR/${model}n.onnx"
        fi
        
        if [ ! -f "$model_file" ]; then
            echo "Model file not found: $model_file"
            continue
        fi
        
        echo "Testing $model..."
        
        for iterations in "${test_iterations[@]}"; do
            # CPU tests
            for threads in "${thread_counts[@]}"; do
                echo "  CPU test: $iterations iterations, $threads threads"
                
                timeout 300 "$SCRIPT_DIR/comprehensive_bench" image "$model" detection "$model_file" "$MODELS_DIR/coco.names" "$DATA_DIR/dog.jpg" --cpu --threads="$threads" --iterations="$iterations" 2>/dev/null | tail -n +2 | while read line; do
                    echo "${timestamp},${line},${CPU_NAME},${GPU_NAME}" >> "$results_file"
                done
                
                # Small delay between tests
                sleep 2
            done
            
            # GPU test (if available)
            if [ "$HAS_GPU" = true ]; then
                echo "  GPU test: $iterations iterations"
                
                timeout 300 "$SCRIPT_DIR/comprehensive_bench" image "$model" detection "$model_file" "$MODELS_DIR/coco.names" "$DATA_DIR/dog.jpg" --gpu --iterations="$iterations" 2>/dev/null | tail -n +2 | while read line; do
                    echo "${timestamp},${line},${CPU_NAME},${GPU_NAME}" >> "$results_file"
                done
                
                sleep 2
            fi
        done
    done
    
    echo "Image benchmarks completed: $results_file"
}

# Function to run video benchmarks
run_video_benchmarks() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_file="$RESULTS_DIR/video_benchmark_${timestamp}.csv"
    
    echo "Running video benchmarks..."
    
    # Check for video file
    local video_file=""
    for ext in mp4 avi mov; do
        video_file=$(find "$DATA_DIR" -name "*.$ext" -type f | head -1)
        if [ -n "$video_file" ]; then
            break
        fi
    done
    
    if [ -z "$video_file" ]; then
        echo "No video file found in $DATA_DIR. Skipping video benchmarks."
        return
    fi
    
    echo "Using video file: $video_file"
    echo "Results will be saved to: $results_file"
    
    # Available models to test
    local models=()
    [ -f "$MODELS_DIR/yolov8n.onnx" ] && models+=("yolo8")
    [ -f "$MODELS_DIR/yolo11n.onnx" ] && models+=("yolo11")
    
    if [ ${#models[@]} -eq 0 ]; then
        echo "No supported models found. Skipping video benchmarks."
        return
    fi
    
    # Create header
    echo "timestamp,model_type,task_type,environment,device,threads,precision,load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,latency_avg_ms,latency_min_ms,latency_max_ms,map_score,frame_count,cpu_name,gpu_name" > "$results_file"
    
    for model in "${models[@]}"; do
        # Handle different naming conventions
        if [ "$model" = "yolo8" ]; then
            local model_file="$MODELS_DIR/yolov8n.onnx"
        else
            local model_file="$MODELS_DIR/${model}n.onnx"
        fi
        
        if [ ! -f "$model_file" ]; then
            continue
        fi
        
        echo "Testing $model with video..."
        
        # CPU test
        echo "  CPU video test"
        timeout 600 "$SCRIPT_DIR/comprehensive_bench" video "$model" detection "$model_file" "$MODELS_DIR/coco.names" "$video_file" --cpu 2>/dev/null | tail -n +2 | while read line; do
            echo "${timestamp},${line},${CPU_NAME},${GPU_NAME}" >> "$results_file"
        done
        
        sleep 5
        
        # GPU test (if available)
        if [ "$HAS_GPU" = true ]; then
            echo "  GPU video test"
            timeout 600 "$SCRIPT_DIR/comprehensive_bench" video "$model" detection "$model_file" "$MODELS_DIR/coco.names" "$video_file" --gpu 2>/dev/null | tail -n +2 | while read line; do
                echo "${timestamp},${line},${CPU_NAME},${GPU_NAME}" >> "$results_file"
            done
            
            sleep 5
        fi
    done
    
    echo "Video benchmarks completed: $results_file"
}

# Function to generate analysis
generate_analysis() {
    echo "Generating analysis reports..."
    
    # Install Python dependencies if needed
    if ! python3 -c "import pandas, matplotlib, seaborn" 2>/dev/null; then
        echo "Installing Python dependencies..."
        pip3 install -r "$SCRIPT_DIR/requirements.txt" --quiet
    fi
    
    # Find the most recent CSV file
    local latest_csv=$(find "$RESULTS_DIR" -name "*.csv" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_csv" ]; then
        echo "No CSV files found for analysis."
        return
    fi
    
    echo "Analyzing: $latest_csv"
    
    # Create analysis output directory
    local analysis_dir="$RESULTS_DIR/analysis_$(date +"%Y%m%d_%H%M%S")"
    mkdir -p "$analysis_dir"
    
    # Run analysis
    if python3 "$SCRIPT_DIR/analyze_results.py" "$latest_csv" --output-dir "$analysis_dir"; then
        echo "Analysis completed: $analysis_dir"
    else
        echo "Analysis failed. Check if Python dependencies are installed."
    fi
}

# Function to create summary
create_summary() {
    local summary_file="$RESULTS_DIR/benchmark_summary_$(date +"%Y%m%d_%H%M%S").md"
    
    cat > "$summary_file" << EOF
# YOLOs-CPP Benchmark Summary

**Date**: $(date)
**Environment**: $CPU_NAME, $CPU_CORES cores, ${TOTAL_MEMORY}MB RAM
**GPU**: $([ "$HAS_GPU" = true ] && echo "$GPU_NAME" || echo "None")

## System Specifications
- **CPU**: $CPU_NAME
- **Cores**: $CPU_CORES
- **Memory**: ${TOTAL_MEMORY}MB
- **GPU**: $([ "$HAS_GPU" = true ] && echo "$GPU_NAME" || echo "CPU Only")

## Benchmark Results

The following files contain the detailed benchmark results:

EOF
    
    # List all result files
    find "$RESULTS_DIR" -name "*.csv" -type f -printf "- %f (%TY-%Tm-%Td %TH:%TM)\n" >> "$summary_file"
    
    cat >> "$summary_file" << EOF

## Key Findings

- Performance measurements include FPS, latency, and resource utilization
- Both CPU and GPU environments tested (where available)
- Multiple iteration counts tested for statistical significance
- System monitoring included during benchmarks

## Files Generated

$(ls -la "$RESULTS_DIR" | tail -n +2)

## Usage

To view detailed analysis, examine the CSV files or run:
\`\`\`bash
python3 benchmark/analyze_results.py results/benchmark_results.csv
\`\`\`

EOF
    
    echo "Summary created: $summary_file"
}

# Main execution
main() {
    echo "Starting automated benchmark system..."
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run checks and setup
    check_dependencies
    detect_environment
    
    # Run benchmarks
    echo "Running comprehensive benchmarks..."
    run_image_benchmarks
    run_video_benchmarks
    
    # Generate analysis and summary
    generate_analysis
    create_summary
    
    echo "=== Benchmark System Complete ==="
    echo "Results available in: $RESULTS_DIR"
    echo ""
    echo "Quick summary:"
    echo "- Environment: $CPU_NAME ($([ "$HAS_GPU" = true ] && echo "with $GPU_NAME" || echo "CPU only"))"
    echo "- Results directory: $RESULTS_DIR"
    echo "- CSV files generated with detailed metrics"
    echo "- Analysis charts and reports created"
}

# Execute main function with all arguments
main "$@"
