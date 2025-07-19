from ultralytics import YOLO
model = YOLO("yolo12n.pt")
model.export(format="onnx", dynamic=True)
