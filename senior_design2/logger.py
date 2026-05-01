import cv2
from datetime import datetime
import os

# Ensure directory exists for flight logs
os.makedirs("logs", exist_ok=True)

def log_event(frame, gps, veg_boxes, line_segments):
    """
    Overlays metadata onto the frame and saves the 'Risk Evidence' to disk.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # Console output for real-time monitoring
    print(f"\n[ALERT] Risk detected at {gps['lat']}, {gps['lng']}")

    # Annotate frame: Draw vegetation detections
    for (x1, y1, x2, y2) in veg_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Annotate frame: Draw detected powerline segments
    for (x1, y1, x2, y2) in line_segments:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Burn-in Metadata Overlay
    y_offset = 30
    metadata = [
        f"Time: {timestamp}",
        f"Lat: {gps['lat']}",
        f"Lng: {gps['lng']}",
        f"Alt: {gps['alt']}"
    ]
    
    for text in metadata:
        cv2.putText(frame, text, (30, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    # Write to local storage
    filename = f"logs/risk_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
