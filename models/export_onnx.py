from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO("yolo11l.pt")

# Export the model to ONNX format
model.export(format="onnx")