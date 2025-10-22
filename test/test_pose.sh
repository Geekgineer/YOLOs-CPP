#! /bin/bash

cd pose

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
./build_test.sh 3

# Run C++ inference pipeline
./build/inference_pose_cpp

# Compare results with google test
./build/compare_pose_results

# ======================================================================================================================

# #! /bin/bash

# # First, fix Windows line endings in this script itself
# sed -i 's/\r$//' "$0"
# set -e

# cd pose

# CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
# echo "Current directory: $CURRENT_DIR"

# # install python dependencies
# python3 -m pip install -r requirements.txt

# # Download test models
# cd models
# ./download_test_models.sh

# # Export models to onnx
# python3 export_onnx_yoloxx.py cpu

# # Build C++ inference pipeline
# cd ../../
# ./build_test.sh 3

# # Run C++ inference pipeline
# ./build/inference_pose_cpp

# # Compare results with google test
# ./build/compare_pose_results

# ======================================================================================================================
#!/bin/bash

# First, fix Windows line endings in this script itself
sed -i 's/\r$//' "$0"

set -e

# Resolve script dir and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Enter pose directory
cd "$SCRIPT_DIR/pose"

# Install python dependencies
python3 -m pip install -r requirements.txt

# Download test models
cd models
./download_test_models.sh

# Export models to onnx (CPU target)
python3 export_onnx_yoloxx.py cpu

# Build C++ inference pipeline (from project root)
cd "$PROJECT_ROOT"
./build_test.sh 3

# Run C++ inference pipeline and comparison
./build/inference_pose_cpp
./build/compare_pose_results