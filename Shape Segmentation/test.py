from ultralytics import YOLO

# Load a model
model = YOLO('shape-segv2.pt')  # load a custom model

# Predict with the model
results = model('test', conf=0.5, save=True)  # predict on an image