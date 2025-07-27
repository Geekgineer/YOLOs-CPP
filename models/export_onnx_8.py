from ultralytics import YOLO
import torch

print("--------------------------------")
print(torch.cuda.is_available())
# Load the YOLOv11n model
model = YOLO("yolov8n.pt")#("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx")