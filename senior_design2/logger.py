# Risk Event Logger with Annotated Image Output
# PURPOSE:
#   When a vegetation encroachment risk is confirmed, this module annotates
#   the triggering frame with visual overlays (powerline segments, vegetation
#   bounding boxes, GPS coordinates, and a timestamp) and saves it as a JPEG
#   to the local "logs/" directory for post-flight review.
#
# WHAT GETS DRAWN ON THE IMAGE:
#   1. Yellow lines: detected powerline segments from powerline_cv.py
#   2. Green boxes: vegetation detections from vegetation_model.py
#   3. Data display: GPS coordinates + timestamp burned into the corner
#
# HOW IT FITS INTO THE SYSTEM:
#   main.py calls log_event() only when risk.py confirms a real encroachment
#   event (or when the operator presses 's' for a manual save). The saved
#   images form the post-flight inspection report for the utility company.
# ============================================================================
 
import cv2
import numpy as np
from datetime import datetime
import os
 
# Create the logs directory if it doesn't already exist.
# exist_ok=True means no error is raised if the folder is already there.
os.makedirs("logs", exist_ok=True)
 
 
def log_event(frame, gps, veg_boxes, line_segments):
    """
    Annotates a risk frame and saves it as a timestamped JPEG.
 
    Parameters:
        frame        (np.ndarray) : BGR image from camera.py at the moment of detection
        gps          (dict)       : {"lat", "lng", "alt"} from gps.py
        veg_boxes    (list)       : [(x1,y1,x2,y2), ...] from vegetation_model.py
        line_segments(list)       : [(x1,y1,x2,y2), ...] from powerline_cv.py
    """
 
    # Generate a timestamp for both the console alert and filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
 
    # Print a console alert for testing purposes to confirm a picture was saved to logs file
    # Would not be in real deployment
    print(f"\n[ALERT] Risk detected at  {round(gps['lat'], 3)}, {round(gps['lng'], 3)}, {gps['alt']}")
 
    # Work on a copy of the frame so the original in memory remains unmodified
    # (the original keeps flowing through the detection pipeline untouched)
    out = frame.copy()
 
    # 1. Powerline segments — drawn as yellow lines
    # Each segment is a straight line detected by Hough transform in powerline_cv.py.
    for (x1, y1, x2, y2) in line_segments:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
 
    # 2. Vegetation bounding boxes — drawn as green rectangles with labels
    BOX_COLOR  = (0, 255, 0)    # Bright green rectangle outline
    LABEL_BG   = (0, 200, 0)    # Slightly darker green filled label background
    LABEL_TEXT = (0, 0, 0)      # Black text for contrast on the green label
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    FONT_THICK = 1
    label      = "vegetation"
 
    # Pre-compute label tag dimensions so the background rectangle fits the text exactly
    (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    pad = 4   # Pixel padding inside the label tag
 
    for (x1, y1, x2, y2) in veg_boxes:
        # Draw the bounding box around the detected vegetation patch
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
 
        # Draw a filled label tag just above the bounding box.
        # max(0, ...) clamps the tag to the frame edge if the box is near the top
        tag_top   = max(0, y1 - lh - pad * 2)
        tag_right = x1 + lw + pad * 2
        cv2.rectangle(out, (x1, tag_top), (tag_right, y1), LABEL_BG, -1)  # -1 = filled
        cv2.putText(out, label, (x1 + pad, y1 - pad),
                    FONT, FONT_SCALE, LABEL_TEXT, FONT_THICK, cv2.LINE_AA)
 

    # 3. GPS / timestamp display
    wsu_gold     = (0, 205, 255)   # WSU gold colored text
    font         = cv2.FONT_HERSHEY_SIMPLEX
    scale        = 0.6
    thickness    = 2
    PANEL_ALPHA  = 0.45            # 0 = fully transparent, 1 = fully opaque
    MARGIN       = 10              # Pixel gap between panel edge and text
    LINE_SPACING = 28              # Vertical pixels between each line of text
 
    # The four data lines burned into the image for post-flight GPS reference
    lines_text = [
        f"Time: {timestamp.replace('_', ' ')}",
        f"Lat: {round(gps['lat'], 3)}",
        f"Lng: {round(gps['lng'], 3)}",
        f"Alt: {gps['alt']}",
    ]
 
    # Size the dark background panel to fit the widest text line exactly
    max_w   = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines_text)
    panel_w = max_w + MARGIN * 2
    panel_h = LINE_SPACING * len(lines_text) + MARGIN
 
    # Alpha blending: draw a solid black rectangle on a copy of the frame,
    # then blend it back at PANEL_ALPHA opacity for the semi-transparent effect
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, out, 1 - PANEL_ALPHA, 0, out)
 
    # Draw each line of text over the blended dark panel
    for i, text in enumerate(lines_text):
        y_pos = MARGIN + (i + 1) * LINE_SPACING - 6
        cv2.putText(out, text, (MARGIN, y_pos),
                    font, scale, wsu_gold, thickness, cv2.LINE_AA)
 
    # 4. Save the annotated frame to disk
    # Filename includes the timestamp so every saved image is uniquely named
    # and can be sorted chronologically during post-flight review
    filename = f"logs/risk_{timestamp}.jpg"
    success  = cv2.imwrite(filename, out)
 
    if success:
        print(f"[SAVED] {filename}")
    else:
        print(f"[ERROR] Failed to write {filename}")
