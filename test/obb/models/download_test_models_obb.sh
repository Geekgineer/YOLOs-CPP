#!/bin/bash

# Download OBB test models

url_base="https://drive.google.com/file/d/15eBSw3gmUNLoYBuMg8hsZTjsaxios39a/view?usp=sharing"
output_dir="."
output_zip="test_models_obb.zip"

# Download the zip file
curl -L "$url_base" -o "$output_zip"

# Extract the zip file
unzip -o "$output_zip" -d "$output_dir"

# Remove the zip file
rm "$output_zip"
