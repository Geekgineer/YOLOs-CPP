import json
import os
import shutil
import sys

import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLOE


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
    if not os.listdir(images_path):
        print(f"No images found in '{images_path}'.")
        return False
    weights_path = paths.get("weights")
    if weights_path is None or not os.path.exists(weights_path):
        print(f"Weights path '{weights_path}' does not exist.")
        return False
    return True


def load_inference_config(config_path: str):
    print(f"Loading inference configuration from '{config_path}'...")
    if not os.path.exists(config_path):
        print(f"Inference configuration file '{config_path}' does not exist.")
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
            print(f"Loaded inference configuration from '{config_path}' : {config} .")
            return config
        except json.JSONDecodeError as e:
            print(f"Error loading inference configuration file '{config_path}': {e}")
            return None


def run_inference(
    model_path: str,
    images_path: str,
    inference_config: dict,
    mask_paths: str,
    class_names: list,
):
    print(
        f"\n ####### Running YOLOE inference for model: {model_path} "
        f"classes={class_names} ... ###### \n"
    )

    model = YOLOE(model_path, task="segment", verbose=True)
    # Exported ONNX already embeds the vocabulary from export; set_classes is PyTorch-only.
    if not model_path.lower().endswith(".onnx"):
        model.set_classes(class_names)

    returned_results = []
    model_name = os.path.basename(model_path).split(".")[0]

    # Sorted for deterministic JSON ordering (must align with C++ inference_yoloe_cpp).
    for image_file in tqdm(sorted(os.listdir(images_path)), desc="Images to process", unit="image"):
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
            device="cpu",
        )

        mask_img_path = os.path.join(mask_paths, f"{model_name}_{image_name}_mask.png")

        if not inference_results or len(inference_results) == 0:
            print(f"No inference results for image '{image_file}', writing empty mask.")
            _im = cv2.imread(image_path)
            if _im is None:
                print(f"ERROR: failed to read image '{image_path}'", file=sys.stderr)
                sys.exit(1)
            h0, w0 = _im.shape[:2]
            full_mask = np.zeros((h0, w0), dtype=np.uint8)
            image_results["mask_path"] = mask_img_path
            cv2.imwrite(mask_img_path, full_mask)
            continue

        result = inference_results[0]
        if result is None:
            print(f"No inference results for image '{image_file}', writing empty mask.")
            _im = cv2.imread(image_path)
            if _im is None:
                print(f"ERROR: failed to read image '{image_path}'", file=sys.stderr)
                sys.exit(1)
            h0, w0 = _im.shape[:2]
            full_mask = np.zeros((h0, w0), dtype=np.uint8)
            image_results["mask_path"] = mask_img_path
            cv2.imwrite(mask_img_path, full_mask)
            continue

        boxes = result.boxes
        if not boxes:
            print(f"No boxes detected for image '{image_file}', writing empty mask.")
            full_mask = np.zeros(result.orig_shape, dtype=np.uint8)
            image_results["mask_path"] = mask_img_path
            cv2.imwrite(mask_img_path, full_mask)
            continue

        class_ids = boxes.cls
        confidences = boxes.conf
        xyxy = boxes.xyxy
        xywh = boxes.xywh

        if result.masks is None:
            print(f"No masks for image '{image_file}', writing empty mask.")
            full_mask = np.zeros(result.orig_shape, dtype=np.uint8)
            image_results["mask_path"] = mask_img_path
            cv2.imwrite(mask_img_path, full_mask)
            continue

        masks_xy = result.masks.xy
        n_det = len(class_ids)
        if len(masks_xy) != n_det or len(xyxy) != n_det or len(xywh) != n_det:
            print(
                f"ERROR: detection tensor length mismatch for '{image_file}' "
                f"(boxes={n_det}, masks={len(masks_xy)})",
                file=sys.stderr,
            )
            sys.exit(1)

        full_mask = np.zeros(result.orig_shape, dtype=np.uint8)

        for mask_xy, class_id, confidence, xyxy_box, xywh_box in zip(
            masks_xy, class_ids, confidences, xyxy, xywh
        ):
            class_id = int(class_id)
            confidence = float(confidence)
            x1, y1, x2, y2 = map(int, xyxy_box)
            x, y, w, h = map(int, xywh_box)
            left, top, width, height = x1, y1, w, h

            image_results["inference_results"].append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                    },
                }
            )

            mask_xy = mask_xy.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(full_mask, [mask_xy], class_id)

        image_results["mask_path"] = mask_img_path
        cv2.imwrite(mask_img_path, full_mask)

    print(f"\n ###### Finished YOLOE inference for model: {model_path} ##### \n")
    return returned_results


def main():
    data_path = "data"
    images_path = os.path.join(data_path, "images")
    weights_path = "models"

    if not validate_paths({"data": data_path, "weights": weights_path}):
        print("Path validation failed, exiting.")
        sys.exit(1)

    results_path = "results"
    if os.path.exists(results_path):
        print(f"Results path '{results_path}' already exists, removing it and creating a new one.")
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    masks_path = os.path.join(results_path, "masks", "ultralytics")
    os.makedirs(masks_path)

    inference_config = {"conf": 0.50, "iou": 0.50}
    inference_config_path = "inference_config.json"
    loaded = load_inference_config(inference_config_path)
    if loaded is not None:
        inference_config["conf"] = float(loaded["conf"])
        inference_config["iou"] = float(loaded["iou"])
        class_names = loaded.get("classes")
    else:
        print(f"Using default inference configuration : {inference_config}.")
        class_names = None

    if not class_names:
        print("ERROR: inference_config.json must define 'classes' for YOLOE parity.")
        sys.exit(1)

    output_results_json = os.path.join(results_path, "results_ultralytics.json")

    models = ["yoloe-26n-seg"]
    results_dict = {}

    for model_name in tqdm(models, desc="Models to test", unit="model"):
        model_weights = os.path.join(weights_path, f"{model_name}.onnx")
        if not os.path.exists(model_weights):
            print(f"Model weights '{model_weights}' do not exist, skipping.")
            continue

        results_dict[model_name] = {
            "weights_path": model_weights,
            "task": "segment",
        }
        model_results = run_inference(
            model_weights, images_path, inference_config, masks_path, class_names
        )
        results_dict[model_name]["results"] = model_results

    if not results_dict:
        print("ERROR: No YOLOE ONNX models found under models/. Export with models/export_yoloe_test_onnx.py")
        sys.exit(1)

    with open(output_results_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to '{output_results_json}'.")
    return


if __name__ == "__main__":
    main()
