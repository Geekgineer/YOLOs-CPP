#! /bin/bash

cd detection

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Current directory: $CURRENT_DIR"

# install python dependencies
python3 -m pip install -r requirements.txt

# Download test models
cd models
./download_test_models.sh

# Export models to onnx
python3 export_onnx_yoloxx.py cpu

# Build C++ inference pipeline
cd ../../
./build_test.sh 0

# Run C++ inference pipeline
./build/inference_detection_cpp

# Compare results with pytest
cd detection
pytest compare_results.py