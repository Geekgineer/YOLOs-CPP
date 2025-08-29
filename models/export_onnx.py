from ultralytics import YOLO

# Load the YOLOv12n model

# model = YOLO("yolov5nu.pt")
# model = YOLO("yolov6n.yaml")
# model.train(data="coco8.yaml", epochs=50, imgsz=640)
# model = YOLO("yolov8n.pt")
# model = YOLO("yolov9t.pt")
# model = YOLO("yolov10n.pt")
model = YOLO("yolo11n.pt")
# model = YOLO("yolo12n.pt")


# Export the model to ONNX format
model.export(format="onnx")