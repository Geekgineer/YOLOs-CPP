from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO("best.pt")

# Export the model to ONNX format
model.export(format="onnx", opset=18)
