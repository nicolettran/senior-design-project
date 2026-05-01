import numpy as np
import time

# ---------------- CONFIGURATION ----------------
RISK_PIXEL_THRESHOLD = 40    # Distance in pixels between tree and wire
MIN_BOX_HEIGHT = 60          # Filter for small shrubs
MIN_BOX_AREA = 4000          # Filter for noise
CONSECUTIVE_FRAMES_REQ = 3   # Persistence filter to avoid false positives
ALERT_COOLDOWN = 10          # Seconds to wait between saving images

# ---------------- GLOBAL STATE ----------------
risk_frame_counter = 0
last_alert_time = 0

def evaluate_risk(veg_boxes, line_segments):
    """
    Determines if vegetation is encroaching on powerlines based on proximity.
    Uses a persistence filter to ensure the risk is real before alerting.
    """
    global risk_frame_counter, last_alert_time
    frame_has_risk = False

    for (x1, y1, x2, y2) in veg_boxes:
        # Size Filtering
        box_height = y2 - y1
        box_area = (x2 - x1) * (y2 - y1)
        if box_height < MIN_BOX_HEIGHT or box_area < MIN_BOX_AREA:
            continue

        # Proximity Check
        for (lx1, ly1, lx2, ly2) in line_segments:
            dist = _point_to_line_dist((x1, y1, x2, y2), (lx1, ly1, lx2, ly2))
            if dist < RISK_PIXEL_THRESHOLD:
                frame_has_risk = True
                break
        if frame_has_risk: break

    # Logic for consecutive detection (Filtering noise)
    if frame_has_risk:
        risk_frame_counter += 1
    else:
        risk_frame_counter = 0

    if risk_frame_counter >= CONSECUTIVE_FRAMES_REQ:
        if (time.time() - last_alert_time) > ALERT_COOLDOWN:
            last_alert_time = time.time()
            risk_frame_counter = 0
            return True
    return False

def _point_to_line_dist(box, line):
    """Calculates shortest distance from bounding box center to a line segment."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x1, y1, x2, y2 = line
    num = abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den if den != 0 else float('inf')
