# Vegetation Encroachment Risk Evaluator
# PURPOSE:
#   The "decision engine" of the system. Given a set of vegetation bounding
#   boxes (from vegetation_model.py) and powerline segments (from powerline_cv.py),
#   this module determines whether any vegetation is dangerously close to a wire.
#
# KEY DESIGN DECISIONS:
#
#   1. PIXEL-SPACE PROXIMITY: Rather than GPS distance (which would require
#      knowing the drone's exact altitude and camera focal length), risk is
#      evaluated in image pixel space. If a vegetation box and a powerline
#      segment are within RISK_PIXEL_THRESHOLD pixels of each other in the
#      frame, that constitutes a potential risk event.
#
#   2. PERSISTENCE FILTER: A single-frame detection is NOT enough to trigger
#      an alert. The risk must appear in CONSECUTIVE_FRAMES_REQ consecutive
#      frames. This eliminates transient false positives caused by motion blur,
#      birds, shadows, or a leaf briefly crossing the camera's field of view.
#
#   3. ALERT COOLDOWN: Once an alert fires and an image is saved, the system
#      waits ALERT_COOLDOWN seconds before saving another. This prevents the
#      logs folder from filling with hundreds of nearly-identical images of the
#      same encroachment site as the drone hovers or moves slowly overhead.
#
#   NOTE: These are all tunable parameters that can be adjusted based on testing
#   and real-world performance. The current values were chosen based on empirical 
#   testing with the drone prototype and can be refined further during field trials.
#   For improvements, working with utility companies such as arborists and line
#   inspectors could help in fine-tuning these parameters and algorithm to better 
#   match real-world risk thresholds and operational needs. These parameters are
#   general heuristics made from client meetings and development testing.
#
# HOW IT FITS INTO THE SYSTEM:
#   main.py calls evaluate_risk(veg_boxes, line_segments) every frame.
#   When it returns True, main.py calls log_event() to save the annotated image.
# ============================================================================
 
import numpy as np
import time
 
# Configuration parameters for risk evaluation — tuned empirically during testing
 
# Maximum pixel distance between a vegetation box center and a powerline segment
# for the pair to be flagged as a potential encroachment risk.
# At typical drone inspection altitude, 300px represents a meaningful proximity.
RISK_PIXEL_THRESHOLD = 300
 
# Ignore vegetation boxes shorter than this many pixels.
# Filters out low-lying ground cover and small shrubs that pose minimal
# risk to distribution lines strung at height.
MIN_BOX_HEIGHT = 50
 
# Ignore vegetation boxes with area smaller than this many square pixels.
# Removes single-pixel noise and tiny artifacts from the ML model's output.
MIN_BOX_AREA = 50
 
# Number of consecutive frames the risk condition must hold before alerting.
# At ~30 FPS, 3 frames ≈ 0.1 seconds — long enough to reject noise,
# short enough to respond to real threats almost instantly.
CONSECUTIVE_FRAMES_REQ = 3
 
# Minimum seconds between saved alert images for the same location.
# Prevents log flooding when the drone is stationary over a risk site.
ALERT_COOLDOWN = 10
 
# Global state variables
# These persist across function calls to implement the persistence filter
# and cooldown timer across frames.
 
risk_frame_counter = 0   # How many consecutive frames have shown a risk condition
last_alert_time    = 0   # Unix timestamp of the last saved alert image
 
 
def evaluate_risk(veg_boxes, line_segments):
    """
    Determines if vegetation is encroaching on powerlines based on proximity.
    Uses a persistence filter to ensure the risk is real before alerting.
 
    Parameters:
        veg_boxes     (list): [(x1,y1,x2,y2), ...] from vegetation_model.py
        line_segments (list): [(x1,y1,x2,y2), ...] from powerline_cv.py
 
    Returns:
        True  — a confirmed, cooled-down risk event; caller should save an image
        False — no risk, or risk present but not yet confirmed / still cooling down
    """
    global risk_frame_counter, last_alert_time
 
    frame_has_risk = False   # Tracks whether THIS frame contains a risk pair
 
    for (x1, y1, x2, y2) in veg_boxes:
 
        # Size Filtering
        # Reject boxes that are too small to represent meaningful vegetation.
        # This guards against ML noise (texture patches, shadow edges, etc.)
        # being mistakenly counted as encroachment risks.
        box_height = y2 - y1
        box_area   = (x2 - x1) * (y2 - y1)
        if box_height < MIN_BOX_HEIGHT or box_area < MIN_BOX_AREA:
            continue   # Skip this box — too small to matter
 
        # Proximity Check 
        # For each remaining vegetation box, measure its distance to every
        # detected powerline segment. If ANY segment is within the risk
        # threshold, this frame is flagged immediately (no need to check more).
        for (lx1, ly1, lx2, ly2) in line_segments:
            dist = _point_to_line_dist((x1, y1, x2, y2), (lx1, ly1, lx2, ly2))
            if dist < RISK_PIXEL_THRESHOLD:
                frame_has_risk = True
                break   # Found one risky pair, no need to check remaining lines
 
        if frame_has_risk:
            break   # Found one risky vegetation box — no need to check remaining boxes
 
    # Persistence Filter
    # Increment the consecutive-frame counter when risk is present this frame,
    # or reset it to zero when this frame is clear.
    # A single clear frame resets the counter, the risk must be continuous.
    if frame_has_risk:
        risk_frame_counter += 1
    else:
        risk_frame_counter = 0   # Risk disappeared, restart the streak
 
    # Alert Decision
    # Only fire an alert if BOTH conditions are met:
    #   1. The risk has persisted for enough consecutive frames (not a fluke)
    #   2. Enough time has passed since the last alert (not a duplicate save)
    if risk_frame_counter >= CONSECUTIVE_FRAMES_REQ:
        if (time.time() - last_alert_time) > ALERT_COOLDOWN:
            last_alert_time    = time.time()   # Record when this alert fired
            risk_frame_counter = 0             # Reset streak so next event needs re-confirmation
            return True                        # Signal main.py to save an annotated image
 
    return False   # Conditions not yet met? Continue monitoring
 
 
def _point_to_line_dist(box, line):
    """
    Calculates the shortest (perpendicular) distance from a bounding box's
    center point to a finite line segment.
 
    This gives a physically meaningful "how close is this tree to the wire?"
    measurement in pixel space.
 
    Parameters:
        box  (tuple): (x1, y1, x2, y2) bounding box corners
        line (tuple): (x1, y1, x2, y2) line segment endpoints
 
    Returns:
        float — perpendicular distance in pixels from the box center to the line.
                Returns infinity if the line segment has zero length (degenerate case).
 
    Math:
        The perpendicular distance from point (cx, cy) to the infinite line
        passing through (x1,y1) and (x2,y2) is:
 
            dist = |( y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1|
                   ─────────────────────────────────────────────
                          sqrt((y2-y1)² + (x2-x1)²)
 
        This is the standard point-to-line distance formula.
        Using perpendicular distance (rather than distance to the nearest endpoint)
        is better here because a wire runs along the full length of the segment,
        vegetation beneath the middle of the wire is just as dangerous as near the ends.
    """
 
    # Compute bounding box center point
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
 
    x1, y1, x2, y2 = line
 
    # Numerator: area of the parallelogram formed by the point and line endpoints
    num = abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
 
    # Denominator: length of the line segment (normalizes the area to a distance)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
 
    # Guard against division by zero for a degenerate zero-length segment
    return num / den if den != 0 else float('inf')
 
