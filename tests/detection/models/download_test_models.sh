#!/bin/bash

# Download test models for detection
# Downloads pre-trained YOLO models fine-tuned on Pascal VOC

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration - Update this URL when creating a new release
REPO_OWNER="Geekgineer"
REPO_NAME="YOLOs-CPP"
RELEASE_TAG="v1.0.0-models"
ASSET_NAME="yolo-detection-models.zip"

url_base="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"
output_zip="${ASSET_NAME}"

echo "Downloading detection test models..."
echo "URL: $url_base"

# Download the zip file
if ! curl -L --fail "$url_base" -o "$output_zip" 2>/dev/null; then
    echo "Warning: Could not download from release. Falling back to Ultralytics download..."
    
    # Fallback: Download directly from Ultralytics
    python3 -c "
from ultralytics import YOLO
import os

models = [
    'yolov5nu.pt', 'yolov6n.pt', 'yolov8n.pt', 'yolov9t.pt',
    'yolov10n.pt', 'yolo11n.pt', 'yolo12n.pt', 'yolo26n.pt'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    try:
        model = YOLO(model_name)
        print(f'  ✓ {model_name} downloaded')
    except Exception as e:
        print(f'  ✗ Failed to download {model_name}: {e}')
"
    echo "Models downloaded via Ultralytics. Remember to fine-tune on VOC dataset."
    exit 0
fi

# Extract the zip file
echo "Extracting models..."
unzip -o "$output_zip" -d "."

# Remove the zip file
rm -f "$output_zip"

echo "Detection models ready!"
