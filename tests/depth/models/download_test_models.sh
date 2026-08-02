#!/bin/bash

# Download test models for depth estimation.
# YOLO26-depth weights come straight from Ultralytics assets.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading depth test models..."

python3 -c "
from ultralytics import YOLO

for model_name in ['yolo26n-depth.pt']:
    print(f'Downloading {model_name}...')
    try:
        YOLO(model_name)
        print(f'  OK {model_name}')
    except Exception as e:
        print(f'  FAILED {model_name}: {e}')
        raise
"

echo "Depth models ready!"
