# Model Guide

Supported models, ONNX export, and optimization for YOLOs-CPP.

## Supported Models

### Detection

| Model | Params | mAP | Speed (GPU) |
|-------|-------:|----:|------------:|
| YOLOv5n | 1.9M | 28.0 | 6.3ms |
| YOLOv8n | 3.2M | 37.3 | 6.2ms |
| YOLOv11n | 2.6M | 39.5 | 6.5ms |
| YOLO26n | 2.5M | 40.2 | 7.1ms |

### Segmentation

| Model | Params | mAP | Speed (GPU) |
|-------|-------:|----:|------------:|
| YOLOv8n-seg | 3.4M | 36.7 | 8.4ms |
| YOLOv11n-seg | 2.9M | 38.9 | 8.1ms |
| YOLO26n-seg | 2.8M | 39.4 | 8.8ms |

### Pose Estimation

| Model | Params | mAP | Speed (GPU) |
|-------|-------:|----:|------------:|
| YOLOv8n-pose | 3.3M | 50.4 | 5.9ms |
| YOLOv11n-pose | 2.9M | 52.1 | 5.7ms |
| YOLO26n-pose | 2.8M | 53.0 | 6.2ms |

### OBB (Oriented Bounding Boxes)

| Model | Params | Dataset |
|-------|-------:|---------|
| YOLOv8n-obb | 3.1M | DOTA |
| YOLOv11n-obb | 2.7M | DOTA |
| YOLO26n-obb | 2.6M | DOTA |

### Classification

| Model | Params | Top-1 Acc |
|-------|-------:|----------:|
| YOLOv8n-cls | 2.7M | 66.6% |
| YOLOv11n-cls | 1.6M | 70.0% |
| YOLO26n-cls | 1.5M | 71.2% |

## Exporting to ONNX

### Using Ultralytics

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    opset=12,        # ONNX opset version
    simplify=False,
    half=False,
    dynamic=False,
    nms=False        # NMS is done in C++
)
```

### Export Options

| Option | Value | Notes |
|--------|-------|-------|
| `opset` | 12-17 | Use 12 for max compatibility |
| `imgsz` | 640 | Match your inference resolution |
| `half` | False | FP32 for accuracy (FP16 optional) |
| `dynamic` | False | Static shapes for best performance |
| `nms` | False | C++ handles NMS |

### Batch Export Script

```bash
python models/export_onnx.py
```

## Label Files

| File | Classes | Use Case |
|------|--------:|----------|
| `coco.names` | 80 | General detection |
| `Dota.names` | 15 | Aerial/satellite OBB |
| `imagenet_classes.txt` | 1000 | Classification |

## Model Paths

```cpp
// Detection
"models/yolo11n.onnx"

// Segmentation
"models/yolo11n-seg.onnx"

// Pose
"models/yolo11n-pose.onnx"

// OBB
"models/yolo11n-obb.onnx"

// Classification
"models/yolo11n-cls.onnx"
```

## Quantization

Quantized models offer:
- **2-4x smaller** file size
- **1.5-2x faster** CPU inference
- **Slight accuracy loss** (~1-2% mAP)

### Quantize with ONNX

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_int8.onnx"
)
```

See `quantized_models/yolos_quantization.py` for examples.

## Custom Models

To use custom-trained models:

1. Train with Ultralytics
2. Export to ONNX with compatible settings
3. Create matching label file
4. Load in YOLOs-CPP

```cpp
yolos::det::YOLODetector detector(
    "custom_model.onnx",
    "custom_labels.txt",
    true
);
```

## Next Steps

- [Usage Guide](usage.md) — API reference
- [Development](development.md) — Extend the library
