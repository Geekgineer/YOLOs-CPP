#!/bin/bash
# ============================================================================
# YOLOs-CPP Classification Test Runner
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP Classification Test"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/classification"
echo "Working directory: $(pwd)"

# Ensure test images exist
print_header "Checking Test Images"
download_test_images "$(pwd)/data/images" "classification"

# Install uv and dependencies
print_header "Installing Dependencies"
install_uv
install_python_packages ultralytics onnx tqdm

# ============================================================================
# Download and Export Models
# ============================================================================
print_header "Preparing Models"
cd models

if [ ! -f "yolo11n-cls.pt" ]; then
    echo "Downloading test models..."
    ./download_test_models.sh 2>/dev/null || {
        echo "Downloading classification models from Ultralytics..."
        python3 -c "
from ultralytics import YOLO
for m in ['yolov8n-cls.pt', 'yolo11n-cls.pt']:
    print(f'Downloading {m}...')
    YOLO(m)
"
    }
fi

export_models_to_onnx "$(pwd)" "export_onnx_yoloxx.py"

# ============================================================================
# Generate Python Ground Truth
# ============================================================================
print_header "Generating Python Ground Truth"
cd "$SCRIPT_DIR/classification"
echo "Running Ultralytics inference..."
python3 inference_classification_ultralytics.py || {
    print_error "Failed to generate Python ground truth"
    exit 1
}
print_success "Python ground truth generated"

# ============================================================================
# Build and Run Tests
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 1

print_header "Running C++ Inference"
cd build
./inference_classification_cpp cpu

print_header "Running Comparison Tests"
./compare_classification_results

print_success "Classification tests completed!"
