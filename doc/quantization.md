# Quantization Guide for YOLOs-CPP

## ⚡ What is Quantization?
Quantization reduces model size and improves inference speed by converting float32 weights to int8 (uint8). This is especially useful for edge devices or real-time performance.

## 🧩 Provided Quantized Models
Quantized versions available in `quantized_models/`:

| Model              | File                         |
|--------------------|------------------------------|
| YOLOv5             | yolo5-n6_uint8.onnx           |
| YOLOv7 Tiny        | yolo7-tiny-uint8.onnx         |
| YOLOv8             | yolo8n_uint8.onnx             |
| YOLOv8 Segmentation| yolo8n-seg_uint8.onnx         |
| YOLOv10            | yolo10n_uint8.onnx            |
| YOLOv11            | yolo11n_uint8.onnx            |
| YOLOv11 Segmentation | yolo11n-seg_uint8.onnx     |

## ⚙️ Custom Quantization
To quantize your own YOLO model:

### 1. Export the ONNX Model
Use the provided script:
```bash
python models/export_onnx.py
```

### 2. Run the Quantization Script
```bash
python quantized_models/yolos_quantization.py --model model.onnx --output model_uint8.onnx
```

### Optional Arguments
- `--calib_data`: Path to calibration images
- `--quant_type`: Type of quantization (e.g., static, dynamic)

## 📝 Tips
- Always validate performance vs. accuracy trade-off
- Test on target hardware (e.g., Jetson, Raspberry Pi)
- Use ONNX Runtime quantization tools for fine-grained control

For help exporting models, see `docs/MODELS.md`.

