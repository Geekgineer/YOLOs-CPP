"""
Depth ground truth from Ultralytics.

Runs the exported ONNX through Ultralytics' own DepthPredictor so the reference
includes its exact preprocessing (LetterBox) and postprocessing
(ops.scale_masks: crop letterbox padding, bilinear rescale to the original size).

Dense float maps do not fit the JSON pattern the other suites use, so each map is
written as raw float32 alongside a JSON index holding its shape and range.
"""

import glob
import json
import os
import shutil
import sys

import numpy as np
from ultralytics import YOLO

DEPTH_SUBDIR = os.path.join("results", "depth", "ultralytics")

# Models compared against Ultralytics. Keep in sync with PARITY_MODELS in
# inference_depth_cpp.cpp and with models/export_onnx_yolo26_depth.py.
PARITY_MODELS = ["yolo26n-depth"]


def run_model(onnx_path: str, image_paths: list, imgsz: int) -> list:
    print(f"\n####### Depth inference for {onnx_path} #######")
    model = YOLO(onnx_path, task="depth")

    entries = []
    for image_path in image_paths:
        results = model.predict(image_path, imgsz=imgsz, device="cpu", verbose=False)
        depth = np.asarray(results[0].depth.data, dtype=np.float32)
        assert depth.ndim == 2, f"expected (H, W) depth, got {depth.shape}"

        stem = os.path.splitext(os.path.basename(image_path))[0]
        model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
        rel = os.path.join(DEPTH_SUBDIR, f"{model_stem}__{stem}.bin")
        depth.tofile(rel)

        entries.append({
            "image_path": os.path.abspath(image_path),
            "depth_file": rel,
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "min": float(depth.min()),
            "max": float(depth.max()),
        })
        print(f"  {stem}: {depth.shape} range {depth.min():.3f}-{depth.max():.3f} m")

    return entries


def main() -> None:
    images_path = os.path.join("data", "images")
    weights_path = "models"

    for path in (images_path, weights_path):
        if not os.path.exists(path):
            print(f"Path '{path}' does not exist.")
            sys.exit(1)

    with open("inference_config.json") as f:
        imgsz = int(json.load(f).get("imgsz", 320))

    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs(DEPTH_SUBDIR)

    image_paths = sorted(
        p for p in glob.glob(os.path.join(images_path, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        print(f"No images found in '{images_path}'.")
        sys.exit(1)

    # Explicit list, like the detection and segmentation harnesses. A glob over
    # models/*.onnx would also pick up the synthetic fixtures that
    # make_synthetic_models.py writes there, including not_depth.onnx, which is
    # deliberately NOT a depth model.
    out = {}
    for model_stem in PARITY_MODELS:
        onnx_path = os.path.join(weights_path, model_stem + ".onnx")
        if not os.path.exists(onnx_path):
            print(f"Skipping {model_stem}: {onnx_path} not found.")
            continue
        out[model_stem] = {
            "weights_path": os.path.abspath(onnx_path),
            "task": "depth",
            "results": run_model(onnx_path, image_paths, imgsz),
        }
    if not out:
        print("No parity models found; nothing to compare.")
        sys.exit(1)

    with open(os.path.join("results", "results_ultralytics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nResults saved to 'results/results_ultralytics.json'.")


if __name__ == "__main__":
    main()
