import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from roboflow import Roboflow
rf = Roboflow(api_key="bkyQpELPqZ5cSfzZlea1")
project = rf.workspace("research-project-sklx9").project("shape-detector-ectj7")
dataset = project.version(8).download("yolov8")