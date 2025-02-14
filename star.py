from ultralytics import YOLO
import sys
import keyboard

model = YOLO('star-obb.pt')         # Choose model to use

for result in model.predict(source=1, save=True, show=True, stream=True, conf=0.75):

    if keyboard.is_pressed('Esc'):  # Hit 'Esc' to exit and close program
        sys.exit(0)
