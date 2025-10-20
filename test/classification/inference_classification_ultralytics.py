import sys
import os
import shutil
import json
from typing import Union
from tqdm.auto import tqdm
from ultralytics import YOLO


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


def run_inference(model_path: str, images_path: str) -> list:
    print(f"\n ####### Running classification for model: {model_path} on images in '{images_path}' ... ###### \n")
    try:
      model = YOLO(model=model_path, task="classify", verbose=True)
    except Exception as e:
      print(f"Skipping model '{model_path}': failed to initialize Ultralytics classifier ({e}).")
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

        inference_results = model.predict(source=image_path, verbose=True, device="cpu")
        if not inference_results or len(inference_results) == 0:
            print(f"No inference results for image '{image_file}', skipping.")
            continue

        probs = inference_results[0].probs
        if probs is None:
            print(f"No probabilities for image '{image_file}', skipping.")
            continue

        top1_id = int(probs.top1)
        top1_conf = float(probs.top1conf)

        image_results["inference_results"].append(
            {
                "class_id": top1_id,
                "confidence": top1_conf
            }
        )

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
    cls_files = [f for f in onnx_files if any(tag in f.lower() for tag in ["cls", "class"]) ]
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


