from ultralytics import YOLO

# Load the YOLOv9s model
model = YOLO("yolov9c-seg.pt")

# Export the model to ONNX format
model.export(format = "onnx")