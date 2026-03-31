""" This code is before conversion of YOLOv11n model to NCNN for better performance on Pi
from ultralytics import YOLO

# Load trained vegetation model
model = YOLO("vegetation_model.pt")

def detect_vegetation(frame):

    results = model(frame, verbose=False)

    boxes = []

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]

            if conf > 0.5:
                boxes.append((
                    int(x1), int(y1),
                    int(x2), int(y2)
                ))

    return boxes 
"""
