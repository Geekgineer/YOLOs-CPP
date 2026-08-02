"""
Classification ground truth using Ultralytics' own preprocessing.

The reference tensor comes from ultralytics.data.augment.classify_transforms(),
i.e. torchvision T.Resize(size, BILINEAR) -> T.CenterCrop(size) -> T.ToTensor()
-> T.Normalize(mean=0, std=1), applied to a PIL image. PIL's bilinear filter is
antialiased, so this is NOT the same as cv2.resize(INTER_LINEAR).

That distinction matters: an earlier version of this script re-implemented the
C++ OpenCV preprocessing instead, which meant the suite compared YOLOs-CPP
against a Python transcription of itself and could not detect a mismatch with
Ultralytics (issue #137).
"""
import sys
import os
import shutil
import json
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Union
from tqdm.auto import tqdm

try:
    from ultralytics.data.augment import classify_transforms
except ImportError:  # pragma: no cover - the test runner installs ultralytics
    classify_transforms = None


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
    Preprocess an image exactly as Ultralytics does for classification.

    Uses ultralytics' own classify_transforms() when available so the reference
    cannot drift from the library under test. The fallback reproduces the same
    torchvision pipeline with PIL directly, for environments without torch.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if classify_transforms is not None:
        arr = classify_transforms(size=target_size)(pil).unsqueeze(0).numpy()
        assert arr.dtype == np.float32, f"expected float32 tensor, got {arr.dtype}"
        return arr

    # torchvision T.Resize(int): shortest edge -> target_size, aspect preserved
    w, h = pil.size
    short, long_ = (w, h) if w <= h else (h, w)
    new_short, new_long = target_size, int(target_size * long_ / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    pil = pil.resize((new_w, new_h), Image.BILINEAR)  # antialiased, like PIL

    # torchvision T.CenterCrop(int): offsets are int(round(diff / 2))
    top = int(round((new_h - target_size) / 2.0))
    left = int(round((new_w - target_size) / 2.0))
    arr = np.asarray(pil)[top:top + target_size, left:left + target_size]

    chw = np.transpose(arr.astype(np.float32) / 255.0, (2, 0, 1))
    out = np.expand_dims(chw, 0)
    assert out.dtype == np.float32, f"expected float32 tensor, got {out.dtype}"
    return out


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
            # Preprocess exactly as Ultralytics does
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
