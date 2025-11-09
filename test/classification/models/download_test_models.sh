#!/bin/bash

# Download test models

url_base="https://github.com/imessam/YOLOs-CPP/releases/download/v0.1-test/test_cls_models.zip"
output_dir="."
output_zip="test_models.zip"

# Download the zip file
curl -L "$url_base" -o "$output_zip"

# Extract the zip file
unzip -o "$output_zip" -d "$output_dir"

# Remove the zip file
rm "$output_zip"