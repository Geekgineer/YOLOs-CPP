from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO("yolov8n.pt")#("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx")