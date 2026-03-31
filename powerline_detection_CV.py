import cv2
import numpy as np

def detect_powerlines(frame):
    # Takes frame from camera and returns a list of line segments [(x1, y1, x2, y2), ...]
    # frame = picam2.capture_array()
    # powerlines = detect_powerlines(frame)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, # Parameters to adjust while testing
        rho=1,  # Distance resolution in pixels
        theta=np.pi / 180,  # Angle resolution in radians
        threshold=100, # Minimum edge pixels to detect a line
        minLineLength=10, # Ignore short segments
        maxLineGap=10 # Allow small gaps for sagging on power lines
    )

    detected_lines = []

    # Filter lines by orientation, nearly horizontal
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle of the line
            angle = abs(np.arctan2(y2 - y1, x2 - x1))

            if angle < np.pi / 6:  # 30 degrees
                detected_lines.append((x1, y1, x2, y2))

    return detected_lines
