#!/bin/bash
# ============================================================================
# YOLOs-CPP YOLOE Segmentation Parity Test (Ultralytics vs C++)
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP YOLOE Segmentation Parity Test"

cd "$SCRIPT_DIR/yoloe"
echo "Working directory: $(pwd)"

print_header "Checking Test Images"
download_test_images "$(pwd)/data/images" "segmentation"

print_header "Installing Dependencies"
install_uv
install_python_packages ultralytics onnx tqdm

print_header "Exporting YOLOE Test ONNX"
cd models
setup_venv
python3 export_yoloe_test_onnx.py || {
    print_error "Failed to export yoloe-26n-seg.onnx (see models/export_yoloe_test_onnx.py)"
    exit 1
}
cd "$SCRIPT_DIR/yoloe"

print_header "Generating Python Ground Truth"
echo "Running Ultralytics YOLOE inference..."
python3 inference_yoloe_ultralytics.py || {
    print_error "Failed to generate Python ground truth"
    exit 1
}
print_success "Python ground truth generated"

print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 6

print_header "Running C++ Inference"
cd build
./inference_yoloe_cpp cpu

print_header "Running Comparison Tests"
./compare_yoloe_results

print_success "YOLOE parity tests completed!"
