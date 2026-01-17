"""
Classification inference using ONNX Runtime with OpenCV preprocessing.
Matches C++ implementation exactly for fair comparison.
"""
import sys
import os
import shutil
import json
import cv2
import numpy as np
import onnxruntime as ort
from typing import Union
from tqdm.auto import tqdm


def validate_paths(paths: dict) -> bool:
    print("Validating paths...")

    data_path = paths.get("data")
    if data_path is None or not os.path.exists(data_path):
        print(f"Data path '{data_path}' does not exist.")
        return False

    images_path = os.path.join(data_path, "images")
    if not os.path.exists(images_path):
        print(f"Images path '{images_path}' does not exist.")
        return False

    images_files = os.listdir(images_path)
    if len(images_files) == 0:
        print(f"No images found in '{images_path}'.")
        return False

    weights_path = paths.get("weights")
    if weights_path is None or not os.path.exists(weights_path):
        print(f"Weights path '{weights_path}' does not exist.")
        return False

    return True


def preprocess_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image using OpenCV to match C++ implementation exactly.
    - Load BGR
    - Convert to RGB
    - Resize shortest side to target_size (using integer division like C++)
    - Center crop to target_size x target_size
    - Normalize to [0, 1]
    - Convert to CHW format
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Resize: shortest side to target_size, using integer division (C++ style)
    if h < w:
        new_h = target_size
        new_w = (w * target_size) // h  # Integer division to match C++
    else:
        new_w = target_size
        new_h = (h * target_size) // w
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center crop
    y_start = max(0, (new_h - target_size) // 2)
    x_start = max(0, (new_w - target_size) // 2)
    cropped = resized[y_start:y_start+target_size, x_start:x_start+target_size]
    
    # Normalize to [0, 1]
    normalized = cropped.astype(np.float32) / 255.0
    
    # Convert HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    batch = np.expand_dims(chw, 0)
    
    return batch


def run_inference(model_path: str, images_path: str) -> list:
    print(f"\n ####### Running classification for model: {model_path} on images in '{images_path}' ... ###### \n")
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        target_size = input_shape[2] if len(input_shape) > 2 else 224
    except Exception as e:
        print(f"Skipping model '{model_path}': failed to load ONNX model ({e}).")
        return []

    returned_results = []

    for image_file in tqdm(os.listdir(images_path), desc="Images to process", unit="image"):
        _, file_ext = os.path.splitext(image_file)
        image_path = os.path.join(images_path, image_file)
        if not os.path.isfile(image_path) or file_ext.lower() not in [".jpg", ".jpeg", ".png"]:
            print(f"Skipping non-image file '{image_file}'.")
            continue

        image_results = {
            "image_path": image_path,
            "inference_results": []
        }
        returned_results.append(image_results)

        try:
            # Preprocess using OpenCV (matches C++ exactly)
            input_tensor = preprocess_image(image_path, target_size)
            
            # Run inference
            outputs = session.run(None, {input_name: input_tensor})
            probs = outputs[0].flatten()
            
            # Get top-1 prediction (model output is already softmax)
            top1_id = int(np.argmax(probs))
            top1_conf = float(probs[top1_id])
            
            image_results["inference_results"].append({
                "class_id": top1_id,
                "confidence": top1_conf
            })
            
        except Exception as e:
            print(f"Error processing image '{image_file}': {e}")
            continue

    print(f"\n ###### Finished classification for model: {model_path} on images in '{images_path}' ... ##### \n")
    return returned_results


def main():
    data_path = "data"
    images_path = os.path.join(data_path, "images")
    weights_path = "models"

    paths_to_validate = {
        "data": data_path,
        "images": images_path,
        "weights": weights_path
    }

    if not validate_paths(paths_to_validate):
        print("Path validation failed, exiting.")
        sys.exit(1)

    results_path = "results"
    if os.path.exists(results_path):
        print(f"Results path '{results_path}' already exists, removing it and creating a new one.")
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    output_results_json = os.path.join(results_path, "results_ultralytics.json")

    # Consider all .onnx files in models dir
    # Prefer classification ONNX models (commonly contain 'cls' or 'class' in name)
    onnx_files = [f for f in os.listdir(weights_path) if f.endswith(".onnx")]
    cls_files = [f for f in onnx_files if any(tag in f.lower() for tag in ["cls", "class"])]
    models = [os.path.splitext(f)[0] for f in (cls_files if len(cls_files) > 0 else onnx_files)]

    results_dict = {}

    for model_name in tqdm(models, desc="Models to test", unit="model"):
        model_weights = os.path.join(weights_path, f"{model_name}.onnx")
        if not os.path.exists(model_weights):
            print(f"Model weights '{model_weights}' do not exist, skipping.")
            continue

        if model_name not in results_dict:
            results_dict[model_name] = {
                "weights_path": model_weights,
                "task": "classify"
            }

        model_results = run_inference(model_weights, images_path)
        results_dict[model_name]["results"] = model_results

    with open(output_results_json, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to '{output_results_json}'.")
    return


if __name__ == "__main__":
    main()
