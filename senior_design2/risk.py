import numpy as np
import time

# ---------------- PARAMETERS ----------------

RISK_PIXEL_THRESHOLD = 40        # proximity in pixels
MIN_BOX_HEIGHT = 60              # ignore very small vegetation
MIN_BOX_AREA = 4000              # ignore tiny shrubs
CONSECUTIVE_FRAMES_REQUIRED = 3  # persistence filter
ALERT_COOLDOWN_SECONDS = 10      # prevent spamming

# ---------------- INTERNAL STATE ----------------

risk_frame_counter = 0
last_alert_time = 0


def evaluate_risk(veg_boxes, line_segments):

    global risk_frame_counter
    global last_alert_time

    frame_has_risk = False

    for (x1, y1, x2, y2) in veg_boxes:

        box_height = y2 - y1
        box_area = (x2 - x1) * (y2 - y1)

        if box_height < MIN_BOX_HEIGHT:
            continue

        if box_area < MIN_BOX_AREA:
            continue

        for (lx1, ly1, lx2, ly2) in line_segments:

            dist = point_to_line_distance(
                (x1, y1, x2, y2),
                (lx1, ly1, lx2, ly2)
            )

            if dist < RISK_PIXEL_THRESHOLD:
                frame_has_risk = True
                break

        if frame_has_risk:
            break

    # Persistence logic
    if frame_has_risk:
        risk_frame_counter += 1
    else:
        risk_frame_counter = 0

    # Confirm risk
    if risk_frame_counter >= CONSECUTIVE_FRAMES_REQUIRED:

        current_time = time.time()

        if current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
            last_alert_time = current_time
            risk_frame_counter = 0
            return True

    return False


def point_to_line_distance(box, line):

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2

    x1, y1, x2, y2 = line

    num = abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    if den == 0:
        return float('inf')

    return num / den