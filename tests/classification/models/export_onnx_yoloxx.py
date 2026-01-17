import sys
import os
from glob import glob
from ultralytics import YOLO
from tqdm.auto import tqdm


def main(device_str: str) -> None:
    device = "mps" if device_str == "mac" else ("0" if device_str == "gpu" else "cpu")

    # Discover local .pt weights; prefer classification checkpoints containing 'cls'
    cwd = os.path.dirname(os.path.abspath(__file__))
    pt_files = sorted(glob(os.path.join(cwd, "*.pt")))
    cls_pt_files = [p for p in pt_files if any(tag in os.path.basename(p).lower() for tag in ["cls", "class"])]
    models_to_export = cls_pt_files if len(cls_pt_files) > 0 else pt_files

    if len(models_to_export) == 0:
        print("No .pt files found to export in current directory.")
        return

    # Classification models typically use 224 input; fall back to 640
    def infer_imgsz_from_name(name: str) -> int:
        base = os.path.basename(name).lower()
        return 224 if ("cls" in base or "class" in base) else 640

    for model_path in tqdm(models_to_export, desc="Exporting models to ONNX"):
        export_configs = {
            "format": "onnx",
            "imgsz": 224 if "cls" in os.path.basename(model_path).lower() else 320,
            "half": False,
            "dynamic": False,
            "simplify": True,
            "opset": 12,  # Use opset 12 for maximum compatibility
            "nms": False,
            "batch": 1,
            "device": device,
        }
        imgsz = export_configs["imgsz"]

        try:
            print(f"Exporting '{os.path.basename(model_path)}' (imgsz={imgsz}) to ONNX on device '{device}' ...")
            model = YOLO(model_path)
            model.export(**export_configs)
            print(f"Successfully exported {model_path} as ONNX.")
        except Exception as e:
            print(f"Failed to export '{model_path}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_onnx_yoloxx.py <device>")
        print("<device>: 'cpu' or 'gpu' or 'mac'")
        sys.exit(1)

    device_arg = sys.argv[1].lower()
    main(device_arg)
   