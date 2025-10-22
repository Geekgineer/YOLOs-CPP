#! /bin/bash

cd classification

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Current directory: $CURRENT_DIR"

# install python dependencies
python -m pip install -r requirements.txt

# Download test models
cd models
./download_test_models.sh

# Export models to onnx
python export_onnx_yoloxx.py cpu

# Build C++ inference pipeline
cd ../../
./build_test.sh 1 ##class

# Run C++ inference pipeline
./build/inference_detection_cpp

# Compare results with google test
./build/compare_detection_results 