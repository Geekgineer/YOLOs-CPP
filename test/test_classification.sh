#! /bin/bash

cd classification

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Current directory: $CURRENT_DIR"

# install python dependencies
python3 -m pip install -r requirements.txt

# Download test models
cd models
./download_test_models.sh

# Export models to onnx
python3 export_onnx_yoloxx.py cpu

# Run Python Ultralytics inference pipeline
cd ..
python3 inference_classification_ultralytics.py

# Build C++ inference pipeline
cd ../
./build_test.sh 1

# Run C++ inference pipeline
./build/inference_classification_cpp

# Compare results with google test
./build/compare_classification_results


