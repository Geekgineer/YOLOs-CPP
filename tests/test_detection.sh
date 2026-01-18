#!/bin/bash
# ============================================================================
# YOLOs-CPP Detection Test Runner
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP Detection Test"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/detection"
echo "Working directory: $(pwd)"

# Ensure test images exist
print_header "Checking Test Images"
download_test_images "$(pwd)/data/images" "detection"

# Install uv and dependencies
print_header "Installing Dependencies"
install_uv
install_python_packages ultralytics onnx tqdm

# ============================================================================
# Download and Export Models
# ============================================================================
print_header "Preparing Models"
cd models

if [ ! -f "YOLOv11n_voc.pt" ]; then
    echo "Downloading test models..."
    ./download_test_models.sh
fi

export_models_to_onnx "$(pwd)" "export_onnx_yoloxx.py"

# ============================================================================
# Generate Python Ground Truth
# ============================================================================
print_header "Generating Python Ground Truth"
cd "$SCRIPT_DIR/detection"
echo "Running Ultralytics inference..."
python3 inference_detection_ultralytics.py || {
    print_error "Failed to generate Python ground truth"
    exit 1
}
print_success "Python ground truth generated"

# ============================================================================
# Build and Run Tests
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 0

print_header "Running C++ Inference"
cd build
./inference_detection_cpp cpu

print_header "Running Comparison Tests"
./compare_detection_results

print_success "Detection tests completed!"
