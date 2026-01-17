import sys
import os
import shutil
import json
import cv2

import numpy as np

from typing import Union
from tqdm.auto import tqdm
from ultralytics import YOLO

def validate_paths(paths : dict) -> bool:

    print("Validating paths...")

    data_path = paths.get("data", None)
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

    weights_path = paths.get("weights", None)
    if weights_path is None or not os.path.exists(weights_path):
        print(f"Weights path '{weights_path}' does not exist.")
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

            print(f"Loaded inference configuration from '{config_path}' : {config} .")

            return config
        
        except json.JSONDecodeError as e:
            print(f"Error loading inference configuration file '{config_path}': {e}")
            return None
        
def run_inference(model_path: str, images_path: str, inference_config: dict, mask_paths: str) -> list:

    print(f"\n ####### Running inference for model: {model_path} on images in '{images_path}' with configuration: {inference_config} ... ###### \n")

    model = YOLO(
        model = model_path,
        task = "segment",
        verbose = True,
    )

    returned_results = []

    model_name = os.path.basename(model_path).split(".")[0]

    for image_file in tqdm(os.listdir(images_path), desc="Images to process", unit="image"):
            
        _, file_ext = os.path.splitext(image_file)

        image_path = os.path.join(images_path, image_file)
        
        if not os.path.isfile(image_path) or file_ext.lower() not in [".jpg", ".jpeg", ".png"]:
            print(f"Skipping non-image file '{image_file}'.")
            continue

        image_name = os.path.splitext(image_file)[0]

        image_results = {
            "image_path": image_path,
            "inference_results": []
        }

        returned_results.append(image_results)
        
        inference_results = model.predict(
            source = image_path,
            conf = inference_config["conf"],
            iou = inference_config["iou"],
            verbose = True,
            device = "cpu",
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

        class_ids = boxes.cls
        confidences = boxes.conf
        xyxy = boxes.xyxy
        xywh = boxes.xywh

        masks_xy = result.masks.xy

        full_mask = np.zeros(result.orig_shape, dtype=np.uint8)

        mask_img_path = os.path.join(mask_paths, f"{model_name}_{image_name}_mask.png")

        for mask_xy, class_id, confidence, xyxy, xywh  in zip(masks_xy, class_ids, confidences, xyxy, xywh):

            class_id = int(class_id)
            confidence = float(confidence)
            x1, y1, x2, y2 = map(int, xyxy)
            x, y, w, h = map(int, xywh)

            left, top, width, height = x1, y1, w, h

            image_results["inference_results"].append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height
                    }
                }
            )

            mask_xy = mask_xy.reshape((-1, 1, 2)).astype(np.int32)

            cv2.fillPoly(full_mask, [mask_xy], class_id)

        image_results["mask_path"] = mask_img_path

        cv2.imwrite(mask_img_path, full_mask)
        

    print(f"\n ###### Finished running inference for model: {model_path} on images in '{images_path}' with configuration: {inference_config} ... ##### \n")            

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

    masks_path = os.path.join(results_path, "masks", "ultralytics")
    if os.path.exists(masks_path):
        print(f"Masks path '{masks_path}' already exists, removing it and creating a new one.")
        shutil.rmtree(masks_path)
    os.makedirs(masks_path)

    inference_config = {
            "conf": 0.50,
            "iou": 0.50
    }

    inference_config_path = "inference_config.json"

    inference_config_loaded = load_inference_config(inference_config_path)

    if inference_config_loaded is not None:
        inference_config = inference_config_loaded
    else:
        print(f"Using default inference configuration : {inference_config}.")

    output_results_json = os.path.join(results_path, "results_ultralytics.json")

    models = [
        "yolov8s-seg",
        "yolov9c-seg",
        "yolo11s-seg",
        "yolo26n-seg"
    ]

    results_dict = {}

    for model_name in tqdm(models, desc="Models to test", unit="model"):

        model_weights = os.path.join(weights_path, f"{model_name}.onnx")
        if not os.path.exists(model_weights):
            print(f"Model weights '{model_weights}' do not exist, skipping.")
            continue

        if model_name not in results_dict:
            results_dict[model_name] = {
                "weights_path": model_weights,
                "task": "segment",
            }

        model_results = run_inference(model_weights, images_path, inference_config, masks_path)

        results_dict[model_name]["results"] = model_results

    with open(output_results_json, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to '{output_results_json}'.")


    return

if __name__ == "__main__":
    main()
        

