#!/bin/bash

# Download test models for classification
# Downloads pre-trained YOLO classification models

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration - Update this URL when creating a new release
REPO_OWNER="Geekgineer"
REPO_NAME="YOLOs-CPP"
RELEASE_TAG="v1.0.0-models"
ASSET_NAME="yolo-classification-models.zip"

url_base="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"
output_zip="${ASSET_NAME}"

echo "Downloading classification test models..."
echo "URL: $url_base"

# Download the zip file
if ! curl -L --fail "$url_base" -o "$output_zip" 2>/dev/null; then
    echo "Warning: Could not download from release. Falling back to Ultralytics download..."
    
    # Fallback: Download directly from Ultralytics
    python3 -c "
from ultralytics import YOLO

models = ['yolov8n-cls.pt', 'yolo11l-cls.pt', 'yolo26n-cls.pt']

for model_name in models:
    print(f'Downloading {model_name}...')
    try:
        model = YOLO(model_name)
        print(f'  ✓ {model_name} downloaded')
    except Exception as e:
        print(f'  ✗ Failed to download {model_name}: {e}')
"
    exit 0
fi

# Extract the zip file
echo "Extracting models..."
unzip -o "$output_zip" -d "."

# Remove the zip file
rm -f "$output_zip"

echo "Classification models ready!"
