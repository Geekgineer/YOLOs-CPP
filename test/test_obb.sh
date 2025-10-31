#!/bin/bash

cd obb

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Current directory: $CURRENT_DIR"

# Install python dependencies
# python3 -m pip install -r requirements.txt

# # Download test models
cd models
# ./download_test_models.sh

# # Export models to onnx
# python3 export_onnx_yolo_obb.py cpu

# Build C++ inference pipeline
cd ../../
./build_test.sh 4

# Run C++ inference pipeline
./build/inference_obb_cpp

# Compare results with google test
./build/compare_obb_results
