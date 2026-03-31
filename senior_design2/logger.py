import cv2
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True) # Creates logs folder

def log_event(frame, gps, veg_boxes, line_segments):

    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    print("\nPossible risk found at:")
    print(f"Time: {timestamp}")
    print(f"Lat: {gps['lat']}")
    print(f"Lng: {gps['lng']}")
    print(f"Alt: {gps['alt']}")

    # Draw vegetation boxes
    for (x1, y1, x2, y2) in veg_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Draw powerlines
    for (x1, y1, x2, y2) in line_segments:
        cv2.line(frame, (x1, y1), (x2, y2), (0,255,255), 2)

    # Overlay timestamp and GPS metadata
    cv2.putText(frame, f"Time: {timestamp}", (30,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Lat: {gps['lat']}", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Lng: {gps['lng']}", (30,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    filename = f"logs/risk_{timestamp}.jpg"
    cv2.imwrite(filename, frame)