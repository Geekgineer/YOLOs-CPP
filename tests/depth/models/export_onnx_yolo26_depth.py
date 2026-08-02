import sys

from ultralytics import YOLO


def main(is_use_gpu=False, is_mac=False):
    device = "mps" if is_mac else ("0" if is_use_gpu else "cpu")

    models_to_export = ["yolo26n-depth.pt"]

    export_configs = {
        "format": "onnx",
        "imgsz": 320,     # smaller input for faster testing
        "half": False,    # FP32
        "dynamic": False, # static shape
        "simplify": True,
        "opset": 12,      # match the other task exports
        "batch": 1,
        "device": device,
    }

    for model_name in models_to_export:
        model = YOLO(model_name)
        model.export(**export_configs)
        print(f"Successfully exported {model_name} as ONNX.")

        # Fail loudly if the export shape is not the dense map the C++ expects.
        import onnx

        onnx_path = model_name.replace(".pt", ".onnx")
        graph = onnx.load(onnx_path).graph
        shape = [d.dim_param or d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
        print(f"  {onnx_path} output0 shape: {shape}")
        assert len(graph.output) == 1, f"expected 1 output, got {len(graph.output)}"
        assert len(shape) == 4 and shape[1] == 1, f"expected [1, 1, H, W], got {shape}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_onnx_yolo26_depth.py <device>")
        print("<device>: 'cpu' or 'gpu' or 'mac'")
        sys.exit(1)

    device_arg = sys.argv[1].lower()
    main(is_use_gpu=device_arg == "gpu", is_mac=device_arg == "mac")
