from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(task = detect, mode=train, epochs = 2, data = "data.yaml", imgsz = 640)