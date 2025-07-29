# Supported Models in YOLOs-CPP

## 🧠 YOLO Variants
This project supports a range of YOLO model versions for detection, segmentation, OBB, and pose estimation.

### ✅ Standard Models
| Purpose         | Model File Name        |
|-----------------|------------------------|
| Detection       | yolo11n.onnx, yolo12n.onnx |
| Segmentation    | yolo11n-seg.onnx, yolo8n-seg.onnx |
| Oriented Detection | yolo11n-obb.onnx     |
| Pose Estimation | yolo11n-pose.onnx, yolov8n-pose.onnx |
| Classification  | (added in 2025.05.15) |

### 🧮 Quantized Models
| Model Type   | Model File Name         |
|--------------|--------------------------|
| YOLOv5       | yolo5-n6_uint8.onnx      |
| YOLOv7       | yolo7-tiny-uint8.onnx    |
| YOLOv8       | yolo8n_uint8.onnx        |
| YOLOv10      | yolo10n_uint8.onnx       |
| YOLOv11      | yolo11n_uint8.onnx       |
| YOLOv11 Seg  | yolo11n-seg_uint8.onnx   |

> Note: Quantized models provide smaller size and faster inference with slight accuracy trade-offs.

## 📂 Labels
- `coco.names`: Class labels for general object detection
- `Dota.names`: Class labels for Oriented Bounding Boxes (OBB)

## 🔁 Exporting ONNX Models
Use the provided script to export from PyTorch:
```bash
python models/export_onnx.py
```

> Tip: Export ONNX models specific to your hardware (e.g. resolution, batch size) for optimal performance.

## 📥 Pretrained Models
Pre-exported models available at:
[Cloud Drive (MEGA)](https://mega.nz/folder/TvgXVRQJ#6M0IZdMOvKlKY9-dx7Uu7Q)

However, exporting your own models is **highly recommended**.

For quantization and export details, see `docs/QUANTIZATION.md`.

