#! /bin/bash

cd test

# install python dependencies
python3 -m pip install -r requirements.txt

# Download test models
cd models
./download_test_models.sh

# Export models to onnx
python3 export_onnx_yoloxx.py cpu

# Build C++ inference pipeline
cd ..
./build_test.sh

# Run C++ inference pipeline
./build/test_inference_cpp

# Compare results with pytest
pytest