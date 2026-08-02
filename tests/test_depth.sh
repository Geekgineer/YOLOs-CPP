#!/bin/bash
# ============================================================================
# YOLOs-CPP Depth Estimation Test Runner
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP Depth Test"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/depth"
echo "Working directory: $(pwd)"

print_header "Checking Test Images"
download_test_images "$(pwd)/data/images" "depth"

print_header "Installing Dependencies"
install_uv
# onnxruntime lets Ultralytics run the exported ONNX for the reference; onnxslim
# is what simplify=True needs. Without them the export only warns, but the
# Ultralytics ONNX prediction fails outright.
install_python_packages ultralytics onnx numpy tqdm onnxruntime onnxslim

# ============================================================================
# Synthetic models for the self-contained tests
# ============================================================================
print_header "Generating Synthetic ONNX Models"
python3 make_synthetic_models.py models || {
    print_error "Failed to generate synthetic models"
    exit 1
}
print_success "Synthetic models generated"

# ============================================================================
# Download and Export Real Models
# ============================================================================
print_header "Preparing Models"
cd models

if [ ! -f "yolo26n-depth.pt" ]; then
    echo "Downloading test models..."
    ./download_test_models.sh
fi

python3 export_onnx_yolo26_depth.py cpu || {
    print_error "Failed to export depth model to ONNX"
    exit 1
}

# ============================================================================
# Generate Python Ground Truth
# ============================================================================
print_header "Generating Python Ground Truth"
cd "$SCRIPT_DIR/depth"
echo "Running Ultralytics inference..."
python3 inference_depth_ultralytics.py || {
    print_error "Failed to generate Python ground truth"
    exit 1
}
print_success "Python ground truth generated"

# ============================================================================
# Build and Run Tests
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 8

print_header "Running C++ Inference"
cd build
./inference_depth_cpp cpu

print_header "Running Comparison Tests"
./compare_depth_results

print_success "Depth tests completed!"
