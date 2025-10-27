#!/bin/bash

# Download test models

url_base="https://github.com/rahmasleam/YOLOs-CPP/releases/download/v0.1-test/yolo-test-models-pose.zip"
output_dir="."
output_zip="yolo-test-models-pose.zip"

# Download the zip file
curl -L "$url_base" -o "$output_zip"

# Extract the zip file
unzip -o "$output_zip" -d "$output_dir"

# Remove the zip file
rm "$output_zip"