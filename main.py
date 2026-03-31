from camera import get_frame
from vegetation_model import detect_vegetation
from powerline_cv import detect_powerlines
from risk import evaluate_risk
from logger import log_event
from gps import get_gps
# All modules tied together: camera, GPS, vegetation detection, powerline CV, risk eval, log

import time

# Runs continuous loop as drone flies assessing risk
def main():
    print("In flight...\n")

    try:
        while True:
            frame = get_frame()

            veg_boxes = detect_vegetation(frame)
            line_segments = detect_powerlines(frame)

            risk_detected = evaluate_risk(veg_boxes, line_segments)

            if risk_detected:
                gps = get_gps()
                log_event(frame, gps, veg_boxes, line_segments)

            time.sleep(0.1) # small delay to stabilize CPU

    except KeyboardInterrupt:
        print("\nFlight terminated. Exiting safely.")

if __name__ == "__main__":
    main()