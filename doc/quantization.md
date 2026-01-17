# Model Quantization Guide

Optimize models for faster inference and smaller file size.

## What is Quantization?

Quantization converts model weights from 32-bit floating point (FP32) to lower precision formats:

| Format | Size | Speed | Accuracy |
|--------|------|-------|----------|
| FP32 | 100% | Baseline | Best |
| FP16 | 50% | 1.5-2x | ~Same |
| INT8 | 25% | 2-4x | -1-2% mAP |

## Quick Start

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "yolo11n.onnx",
    "yolo11n_int8.onnx"
)
```

## Dynamic Quantization

Best for **CPU inference**. No calibration data needed.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="yolo11n.onnx",
    model_output="yolo11n_int8.onnx",
    weight_type=QuantType.QUInt8
)
```

## Static Quantization

Better accuracy with **calibration data**.

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class YOLOCalibrationReader(CalibrationDataReader):
    def __init__(self, images_dir, input_name, input_shape):
        self.images = [...]  # Load calibration images
        self.input_name = input_name
        self.input_shape = input_shape
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        # Preprocess image
        data = self.preprocess(self.images[self.index])
        self.index += 1
        return {self.input_name: data}

calibration_reader = YOLOCalibrationReader(
    "calibration_images/",
    "images",
    [1, 3, 640, 640]
)

quantize_static(
    "yolo11n.onnx",
    "yolo11n_static_int8.onnx",
    calibration_reader
)
```

## Using Quantized Models

```cpp
// Same API as FP32 models
yolos::det::YOLODetector detector(
    "yolo11n_int8.onnx",
    "coco.names",
    false  // CPU (quantized models are CPU-optimized)
);

auto detections = detector.detect(frame);
```

## Benchmarks

Tested on Intel i7-12700H (CPU):

| Model | Size | Latency | mAP |
|-------|-----:|--------:|----:|
| YOLOv11n (FP32) | 5.4MB | 67ms | 39.5 |
| YOLOv11n (INT8) | 1.8MB | 28ms | 38.2 |
| YOLOv8n (FP32) | 6.2MB | 72ms | 37.3 |
| YOLOv8n (INT8) | 2.1MB | 31ms | 36.1 |

## Tips

1. **Calibration data matters** — Use 100-500 representative images
2. **Test accuracy** — Validate mAP after quantization
3. **CPU only** — INT8 is optimized for CPU, not GPU
4. **Per-channel** — Better accuracy than per-tensor

## Script

See `quantized_models/yolos_quantization.py` for a complete example.

## Next Steps

- [Model Guide](models.md) — ONNX export
- [Benchmarks](../benchmarks/README.md) — Performance testing
