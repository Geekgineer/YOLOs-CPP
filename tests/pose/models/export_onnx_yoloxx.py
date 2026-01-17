import sys

from ultralytics import YOLO
from tqdm.auto import tqdm

def main(is_use_gpu = False, is_mac = False):

    print(f"Exporting to {device_arg} ...")

    device = "mps" if is_mac else ("0" if is_use_gpu else "cpu")

    models_to_export = [
        "yolov8n-pose.pt",
        "yolo11n-pose.pt",
        "yolo26n-pose.pt"
    ]

    export_configs = {
        "format": "onnx",
        "imgsz": 320,  # Use smaller size for faster testing
        "half": False, # Use FP32
        "dynamic": False, # Use static shape, if dynamic is True, it may cause issues with some runtimes.
        "simplify": True,
        "opset": 12,  # Use opset 12 for maximum compatibility
        "nms": False,
        "batch": 1,
        "device": device
    }

    for model_name in tqdm(models_to_export, desc="Exporting models to ONNX"):

        model = YOLO(model_name)
        model.export(**export_configs)
        
        print(f"Successfully exported {model_name} as ONNX.")

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python export_onnx_yoloxx.py <device>")
        print("<device>: 'cpu' or 'gpu' or 'mac'")

        sys.exit(1)
        
    device_arg = sys.argv[1].lower()

    main(is_use_gpu = device_arg == "gpu", is_mac = device_arg == "mac")
   