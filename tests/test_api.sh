#!/bin/bash
# ============================================================================
# YOLOs-CPP API Test Runner (batch inference + in-memory model loading)
# ============================================================================
# Self-contained: generates tiny synthetic ONNX models instead of downloading
# real weights, so it needs only `onnx` and `numpy` (no PyTorch/Ultralytics).
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP API Test (batch inference + in-memory loading)"

# ============================================================================
# Generate Synthetic Models
# ============================================================================
print_header "Installing Dependencies"
install_uv
install_python_packages onnx numpy

print_header "Generating Synthetic ONNX Models"
cd "$SCRIPT_DIR/api"
python3 make_synthetic_models.py models || {
    print_error "Failed to generate synthetic models"
    exit 1
}
print_success "Synthetic models generated"

# ============================================================================
# Build and Run Tests
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 7

print_header "Running API Tests"
cd build
./test_api_batch_and_memory

print_success "API tests completed!"
