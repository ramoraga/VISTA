from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt') # load a pretrained model

# Train the model
results = model.train(data='datasets\Shape-Detector-8\data.yaml', epochs=150, imgsz=640, batch=8)