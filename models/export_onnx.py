from ultralytics import YOLO
from ultralytics import NAS

# Load the YOLO model
# model = YOLO("yolov5nu.pt")
# model = YOLO("yolov6n.yaml")
# model.train(data="coco8.yaml", epochs=50, imgsz=640)
# model = YOLO("yolov7n.pt")
# model = YOLO("yolov8n.pt")
# model = YOLO("yolov9t.pt")
# model = YOLO("yolov10n.pt")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo12n.pt")
model = NAS("yolo_nas_s.pt")

# Export the model to ONNX format with static batch size (default)
# model.export(format="onnx")

# model = YOLO("yolo11n.pt")
model.export(format="onnx", dynamic=True, opset=11)

# Example: Export with dynamic batch size for batch inference support
# Uncomment the following lines to export a dynamic model
# This will create an ONNX model with dynamic batch dimension (batch size = -1)
# Useful for batch processing multiple images at once

