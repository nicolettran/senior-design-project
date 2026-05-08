# Traditional Computer Vision Powerline Detector
# PURPOSE:
#   Detects powerline wire segments in a camera frame using classical computer
#   vision techniques, NO machine learning involved. The pipeline converts
#   the frame to grayscale, smooths it, finds edges, then uses the Probabilistic
#   Hough Transform to find long, nearly-horizontal or nearly-vertical lines
#   that match the expected geometry of distribution line wires.
#
# WHY TRADITIONAL CV INSTEAD OF ML HERE?
#   Powerlines have a highly predictable visual signature: long, thin, straight
#   lines oriented close to horizontal. Classical edge detection + Hough
#   transform exploits this geometry directly and runs extremely fast on the
#   Raspberry Pi's CPU. No GPU or inference engine required. The ML model
#   (vegetation_model.py) handles the less predictable, organic shapes of trees.
#
# PIPELINE OVERVIEW:
#   Frame → Grayscale → Bilateral Filter → Canny Edges →
#   Hough Lines → Angle Filter → Merge Duplicates → Top-N by Length
#
# HOW IT FITS INTO THE SYSTEM:
#   main.py calls detect_powerlines(frame) each loop iteration.
#   The returned list of line segments is passed to risk.py (proximity check)
#   and logger.py (drawing yellow overlays on saved images).
# ============================================================================

import cv2
import numpy as np


def detect_powerlines(frame):
    """
    Detects powerline segments in a single BGR frame.

    Returns:
        List of (x1, y1, x2, y2) tuples representing detected wire segments,
        sorted by length (longest first), capped at MAX_LINES results.
    """

    h, w = frame.shape[:2]   # Frame height and width in pixels

    # Detection parameters
    # A line must span at least 30% of the frame width to qualify as a wire.
    # Short blobs (leaves, branches, poles) will be ignored by this filter.
    MIN_LINE_LENGTH = int(w * 0.30)

    # Maximum pixel gap allowed between two collinear edge points before
    # Hough treats them as separate lines. 40px bridges gaps at utility poles.
    MAX_GAP = 40

    # Two detected segments whose midpoints are within 30px of each other
    # and point in the same direction get merged into one representative line
    MERGE_DISTANCE = 30

    # Lines within 15° of horizontal (0°) or vertical (90°) are kept.
    # Powerlines are always close to horizontal; this rejects diagonal noise.
    ANGLE_TOLERANCE = 15   # degrees

    # Hard cap on returned results — only keep the longest 8 lines to avoid
    # flooding the risk evaluator with noise from background structures
    MAX_LINES = 8

    # Step 1: Grayscale Conversion
    # Edge detection operates on intensity gradients, not color, so we reduce
    # the 3-channel BGR frame to a single luminance channel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Bilateral Filter (Noise Reduction)
    # Standard Gaussian blur would smooth edges as well as noise.
    # Bilateral filter preserves sharp edges (like wire silhouettes against sky)
    # while smoothing out texture noise from foliage and ground clutter.
    #   d=11         : pixel neighborhood diameter — larger = more smoothing
    #   sigmaColor=90: intensity difference threshold for edge preservation
    #   sigmaSpace=90: spatial smoothing range
    blurred = cv2.bilateralFilter(gray, d=11, sigmaColor=90, sigmaSpace=90)

    # Step 3: Canny Edge Detection
    # Canny finds pixels where intensity changes sharply (i.e., wire edges).
    #   threshold1=60 : weak edges below this are discarded
    #   threshold2=180: strong edges above this are always kept;
    #                   weak edges (60–180) are kept only if connected to strong ones
    #   apertureSize=3: Sobel kernel size used to compute gradients
    edges = cv2.Canny(blurred, 60, 180, apertureSize=3)

    # Step 4: Probabilistic Hough Line Transform
    # Hough transform "votes" for lines that pass through many edge pixels.
    # The probabilistic variant returns actual endpoint coordinates (x1,y1,x2,y2)
    # rather than infinite lines, making it directly usable for drawing and math.
    #   rho=1           : line position resolution (1 pixel)
    #   theta=π/180     : line angle resolution (1 degree)
    #   threshold=120   : minimum votes required — high value means high confidence,
    #                     reduces false positives from short or noisy edges
    #   minLineLength   : segments shorter than this are rejected (see above)
    #   maxLineGap      : gaps smaller than this are bridged (see above)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_GAP
    )

    if lines is None:
        return []   # No lines detected in this frame — return empty list

    # Step 5: Angle Filter
    # Hough will find any long line, including diagonal fences, tree trunks, etc.
    # We keep only lines whose angle is within ANGLE_TOLERANCE of 0° (horizontal)
    # or 90° (vertical), which covers horizontal wires and vertical utility poles.
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1

        # Skip degenerate zero-length segments
        if dx == 0 and dy == 0:
            continue

        # arctan2 returns angle in [-180°, 180°]; % 90 collapses to [0°, 90°]
        # so both +45° and -45° map to 45°, and both polarities of horizontal
        # (left-to-right or right-to-left) map correctly to ~0°
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 90

        # Keep if within ANGLE_TOLERANCE of 0° (horizontal) or 90° (vertical)
        if angle <= ANGLE_TOLERANCE or angle >= (90 - ANGLE_TOLERANCE):
            filtered.append((x1, y1, x2, y2))

    # Step 6: Merge Nearby Duplicate Segments
    # Hough often returns 3–5 nearly identical segments for the same physical wire
    # (one per distinct group of edge pixels along the wire). We merge segments
    # that are close in space and angle into a single representative segment.
    merged = _merge_close_lines(filtered, MERGE_DISTANCE)

    # Step 7: Sort by Length, Return Top N
    # Longer segments are more likely to be actual wires than short noise artifacts.
    # np.hypot computes Euclidean length from endpoint coordinates.
    merged.sort(key=lambda s: np.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)
    return merged[:MAX_LINES]


def _merge_close_lines(segments, threshold):
    """
    Merges line segments that are spatially close and nearly parallel.
    Prevents the same physical wire from producing many redundant detections.

    Strategy:
        For each unmerged segment, find all other segments whose midpoints
        are within `threshold` pixels AND whose angles differ by less than 15°.
        From that group, keep only the longest segment as the representative.

    Parameters:
        segments  (list) : list of (x1, y1, x2, y2) tuples from the angle filter
        threshold (float): max midpoint-to-midpoint distance to consider a merge

    Returns:
        Deduplicated list of (x1, y1, x2, y2) tuples
    """
    if not segments:
        return []

    used   = [False] * len(segments)   # Tracks which segments have been consumed
    merged = []

    for i, (x1, y1, x2, y2) in enumerate(segments):
        if used[i]:
            continue   # Already absorbed into a previous merge group — skip

        # Midpoint and angle of the current "anchor" segment
        mx_i  = (x1 + x2) / 2
        my_i  = (y1 + y2) / 2
        ang_i = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 90

        # Start with the anchor as the best (longest) candidate
        best     = (x1, y1, x2, y2)
        best_len = np.hypot(x2 - x1, y2 - y1)

        for j, (ax1, ay1, ax2, ay2) in enumerate(segments):
            if i == j or used[j]:
                continue

            # Midpoint and angle of the candidate segment
            mx_j  = (ax1 + ax2) / 2
            my_j  = (ay1 + ay2) / 2
            ang_j = abs(np.degrees(np.arctan2(ay2 - ay1, ax2 - ax1))) % 90

            # Euclidean distance between midpoints
            dist = np.hypot(mx_i - mx_j, my_i - my_j)

            # Merge if close enough AND pointing in roughly the same direction
            if dist < threshold and abs(ang_i - ang_j) < 15:
                used[j] = True   # Mark candidate as consumed
                l = np.hypot(ax2 - ax1, ay2 - ay1)

                # Keep whichever segment in this group is longest
                if l > best_len:
                    best     = (ax1, ay1, ax2, ay2)
                    best_len = l

        used[i] = True       # Mark anchor as consumed
        merged.append(best)  # Add the longest representative to the output

    return merged
