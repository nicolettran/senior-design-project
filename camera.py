from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720)}  # Adjust frame size as needed
)
picam2.configure(config)
picam2.start()

def get_frame():
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Converts to BGR for OpenCV
    return frame