# System Entry Point & Main Flight Loop
# PURPOSE:
#   Ties together all five subsystems into a single real-time inspection loop.
#   On every iteration it: grabs a GPS-tagged frame, runs ML vegetation
#   detection, runs CV powerline detection, evaluates encroachment risk, and
#   saves an annotated alert image if a real risk is confirmed.
#
# SYSTEM ARCHITECTURE (data flow each loop iteration):
#
#   CameraStream ──► frame + gps
#                        │
#             ┌──────────┴──────────┐
#             ▼                     ▼
#   detect_vegetation()    detect_powerlines()
#    (ML, NCNN YOLOv8)    (Classical CV, Hough)
#             │                     │
#             └──────────┬──────────┘
#                        ▼
#                  evaluate_risk()
#                        │
#                   risk == True?
#                        │
#                        ▼
#                   log_event()  ──► logs/risk_<timestamp>.jpg
#
# CONTROLS:
#   's' key ──► manual save: immediately logs the current frame regardless of
#             automated risk detection (useful for operator-spotted anomalies)
#   Ctrl+C  ──► graceful shutdown: stops camera thread and exits cleanly
# ============================================================================

from camera           import CameraStream       # Threaded camera + GPS bundle
from vegetation_model import detect_vegetation  # ML model: finds vegetation boxes
from powerline_cv     import detect_powerlines  # CV algorithm: finds wire segments
from risk             import evaluate_risk      # Decision engine: proximity + persistence
from logger           import log_event          # Annotates and saves alert images
import time
import cv2


def main():
    print("Initializing drone inspection system...")

    # System Startup 
    # CameraStream() launches its background thread immediately on construction.
    # The GPS singleton (imported inside camera.py) also starts its serial
    # read thread at import time. Both are running by the time we reach the loop.
    stream = CameraStream()

    # Allow 2 seconds for:
    #   - Camera to warm up and produce stable frames (first frames can be dark)
    #   - GPS module to parse its first NMEA sentences from the serial port
    #   - Autofocus to find a stable focus distance
    time.sleep(2)

    print("In flight...")

    try:
        while True:
            # Step 1: Acquire Data Bundle
            # get_latest() is a fast, non-blocking memory read.
            # The camera background thread has already captured and cached the
            # most recent frame+GPS pair — we just retrieve it here.
            frame, gps = stream.get_latest()

            # Skip this iteration if the camera hasn't produced a frame yet
            # (can happen during the first few milliseconds after startup)
            if frame is None:
                continue

            # Step 2: Run Detection Modules
            # Both detectors operate independently on the same frame.

            # ML model: finds vegetation patches and returns bounding boxes
            veg_boxes = detect_vegetation(frame)

            # Classical CV: finds wire segments using Hough transform
            line_segments = detect_powerlines(frame)

            # Live Preview Window
            # Shows the raw (unannotated) camera feed to the operator.
            # Annotated images are only saved to disk — not shown live —
            # to keep the display loop as fast as possible.
            # In actual deployment, this video stream is NOT shown 
            # — it's only for development debugging and testing purposes.
            cv2.imshow("Drone View", frame)
            key = cv2.waitKey(1) & 0xFF

            # Manual save, press key 's' to capture the current frame
            # with all detections overlaid, regardless of automated risk status
            # Only for testing purposes 
            # in real deployment, the operator would not have this control
            if key == ord('s'):
                print(f" MANUAL SAVE TRIGGERED at GPS: {gps['lat']}, {gps['lng']}, {gps['alt']}")
                log_event(frame, gps, veg_boxes, line_segments)

            # Step 3: Risk Assessment
            # Passes both detection outputs to the risk evaluator.
            # evaluate_risk() applies size filtering, proximity math,
            # a persistence filter (N consecutive frames), and a cooldown timer.
            # Returns True only when all conditions are met simultaneously.
            risk_detected = evaluate_risk(veg_boxes, line_segments)

            # Debug print: Only for development testing, 
            # in actual deployment this would be removed or replaced
            # Shows how many objects were detected and whether risk was flagged
            print(f"[RISK] veg={len(veg_boxes)} lines={len(line_segments)} risk={risk_detected}")

            # Step 4: Log Confirmed Risk Events
            # When risk is confirmed, log_event() draws all overlays onto the
            # frame and saves it as a timestamped JPEG in the logs/ directory.
            # The GPS data used here comes from the exact moment the frame was
            # captured (bundled atomically in CameraStream._update()), ensuring
            # the saved image is accurately geotagged.
            if risk_detected:
                log_event(frame, gps, veg_boxes, line_segments)

            # Step 5: Yield CPU to Background Threads
            # A 10ms sleep gives the camera thread and GPS thread time to run
            # between main loop iterations, preventing CPU starvation that
            # would slow down frame capture or GPS parsing.
            time.sleep(0.01)

    except KeyboardInterrupt:
        # Graceful Shutdown (Ctrl+C)
        # stop() signals the camera background thread to exit its loop,
        # waits for it to finish, and releases the camera hardware.
        # The GPS thread is a daemon and exits automatically with the process.
        print("\nTerminating flight threads...")
        stream.stop()

if __name__ == "__main__":
    main()
