import sys
import os
import shutil
import json
import cv2
import numpy as np
from typing import Union
from tqdm.auto import tqdm
from ultralytics import YOLO

def validate_paths(paths: dict) -> bool:
    print("Validating paths...")
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"{key} path '{path}' does not exist.")
            return False
        if key == "images":
            images_files = os.listdir(path)
            if len(images_files) == 0:
                print(f"No images found in '{path}'.")
                return False
    return True

def load_inference_config(config_path: str) -> Union[dict, None]:
    print(f"Loading inference configuration from '{config_path}'...")
    if not os.path.exists(config_path):
        print(f"Inference configuration file '{config_path}' does not exist.")
        return None
    with open(config_path, "r") as f:
        try:
            config = json.load(f)
            print(f"Loaded inference configuration from '{config_path}' : {config}.")
            return config
        except json.JSONDecodeError as e:
            print(f"Error loading inference configuration file '{config_path}': {e}")
            return None

def run_inference(model_path: str, images_path: str, inference_config: dict) -> list:

    print(f"\n ####### Running inference for model: {model_path} on images in '{images_path}' with configuration: {inference_config} ... ###### \n")

    model = YOLO(model=model_path, task="pose", verbose=True)

    returned_results = []

    model_name = os.path.basename(model_path).split(".")[0]

    for image_file in tqdm(os.listdir(images_path), desc="Images to process", unit="image"):

        _, file_ext = os.path.splitext(image_file)

        image_path = os.path.join(images_path, image_file)

        if not os.path.isfile(image_path) or file_ext.lower() not in [".jpg", ".jpeg", ".png"]:
            print(f"Skipping non-image file '{image_file}'.")
            continue

        image_name = os.path.splitext(image_file)[0]
        image_results = {"image_path": image_path, "inference_results": []}

        returned_results.append(image_results)

        inference_results = model.predict(
            source=image_path,
            conf=inference_config["conf"],
            iou=inference_config["iou"],
            verbose=True,
            device="cpu"
        )

        if not inference_results or len(inference_results) == 0:
            print(f"No inference results for image '{image_file}', skipping.")
            continue

        result = inference_results[0]

        if result is None:
            print(f"No inference results for image '{image_file}', skipping.")
            continue

        boxes = result.boxes

        if not boxes:
            print(f"No boxes detected for image '{image_file}', skipping.")
            continue

        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        keypoints_data = result.keypoints.data.cpu().numpy()

        for idx in range(len(class_ids)):

            kps = []

            for kp in keypoints_data[idx]:
                kps.append({"x": float(kp[0]), "y": float(kp[1]), "confidence": float(kp[2])})

            class_id = int(class_ids[idx])
            confidence = float(confidences[idx])
            x1, y1, x2, y2 = map(int, xyxy[idx])
            x, y, w, h = map(int, xywh[idx])

            image_results["inference_results"].append({
                "class_id": class_id,
                "confidence": confidence,
                "bbox": {"left": x1, "top": y1, "width": w, "height": h},
                "keypoints": kps
            })
            
    print(f"\n ###### Finished running inference for model: {model_path} on images in '{images_path}' with configuration: {inference_config} ... ##### \n")
   
    return returned_results

def main():
    # base_path = os.path.dirname(__file__)
    data_path = "data" #os.path.join(base_path, "data")
    images_path =  os.path.join(data_path, "images")
    weights_path = "models" #os.path.join(base_path, "models")
    results_path = "results" # os.path.join(base_path, "results")
    paths_to_validate = {
        "data": data_path,
        "images": images_path,
        "weights": weights_path,
        "results": results_path
    }
    if not validate_paths(paths_to_validate):
        print("Path validation failed, exiting.")
        sys.exit(1)
    if os.path.exists(results_path):
        print(f"Results path '{results_path}' already exists, removing it and creating a new one.")
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    inference_config = {"conf": 0.50, "iou": 0.50}
    inference_config_path = "inference_config.json" # os.path.join(base_path, "inference_config.json")
    inference_config_loaded = load_inference_config(inference_config_path)
    if inference_config_loaded is not None:
        inference_config = inference_config_loaded
    else:
        print(f"Using default inference configuration : {inference_config}.")
    output_results_json = os.path.join(results_path, "results_ultralytics.json")
    models = ["yolov8n-pose", "yolo11n-pose", "yolo26n-pose"]
    results_dict = {}
    for model_name in tqdm(models, desc="Models to test", unit="model"):
        model_weights = os.path.join(weights_path, f"{model_name}.onnx")
        if not os.path.exists(model_weights):
            print(f"Model weights '{model_weights}' do not exist, skipping.")
            continue
        if model_name not in results_dict:
            results_dict[model_name] = {"weights_path": model_weights, "task": "pose"}
        model_results = run_inference(model_weights, images_path, inference_config)
        results_dict[model_name]["results"] = model_results
    with open(output_results_json, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to '{output_results_json}'.")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    main()