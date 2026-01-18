#!/bin/bash
# ============================================================================
# YOLOs-CPP OBB Test Runner
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP OBB Test"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/obb"
echo "Working directory: $(pwd)"

# Ensure test images exist
print_header "Checking Test Images"
download_test_images "$(pwd)/data/images" "obb"

# Install uv and dependencies
print_header "Installing Dependencies"
install_uv
install_python_packages ultralytics onnx tqdm

# ============================================================================
# Download and Export Models
# ============================================================================
print_header "Preparing Models"
cd models

if [ ! -f "yolo11n-obb.pt" ] && [ ! -f "yolov8n-obb.pt" ]; then
    echo "Downloading test models..."
    ./download_test_models_obb.sh 2>/dev/null || {
        echo "Downloading OBB models from Ultralytics..."
        python3 -c "
from ultralytics import YOLO
for m in ['yolov8n-obb.pt', 'yolo11n-obb.pt']:
    print(f'Downloading {m}...')
    YOLO(m)
"
    }
fi

export_models_to_onnx "$(pwd)" "export_onnx_yolo_obb.py"

# ============================================================================
# Generate Python Ground Truth
# ============================================================================
print_header "Generating Python Ground Truth"
cd "$SCRIPT_DIR/obb"
echo "Running Ultralytics inference..."
python3 inference_obb_ultralytics.py || {
    print_error "Failed to generate Python ground truth"
    exit 1
}
print_success "Python ground truth generated"

# ============================================================================
# Build and Run Tests
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 4

print_header "Running C++ Inference"
cd build
./inference_obb_cpp cpu

print_header "Running Comparison Tests"
./compare_obb_results

print_success "OBB tests completed!"
