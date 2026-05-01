from camera import CameraStream
from vegetation_model import detect_vegetation
from powerline_cv import detect_powerlines
from risk import evaluate_risk
from logger import log_event
import time

def main():
    print("Initializing Asynchronous Drone Inspection System...")
    stream = CameraStream()
    
    # Warm-up time for camera and GPS lock
    time.sleep(2)
    
    print("In flight... ML processing is active.")

    try:
        while True:
            # 1. Grab latest data bundle (Zero hardware latency here)
            frame, gps = stream.get_latest()

            if frame is None:
                continue

            # 2. Parallel AI and CV Processing
            # Note: These are independent modules for vegetation and lines
            veg_boxes = detect_vegetation(frame)
            line_segments = detect_powerlines(frame)

            # 3. Analyze spatial relationship (Risk Assessment)
            risk_detected = evaluate_risk(veg_boxes, line_segments)

            if risk_detected:
                # Use the GPS bundle from the exact moment the frame was taken
                log_event(frame, gps, veg_boxes, line_segments)

            # Small sleep to yield CPU time for background threads
            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\nTerminating flight threads...")
        stream.stop()

if __name__ == "__main__":
    main()
