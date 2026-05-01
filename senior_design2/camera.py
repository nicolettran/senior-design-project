from picamera2 import Picamera2
import cv2
import threading
from gps import gps_module

class CameraStream:
    """
    Maintains a high-speed camera thread that 'bundles' GPS data with frames.
    This ensures that the image and the coordinates are synced in time.
    """
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (1280, 720)})
        self.picam2.configure(config)
        self.picam2.start()

        self.frame = None
        self.gps_data = None
        self.running = True
        self.lock = threading.Lock() # Ensures data integrity between threads

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            # Capture from camera buffer
            raw_frame = self.picam2.capture_array()
            
            # Instantly grab GPS from the GPS thread's memory
            current_gps = gps_module.get_last_known_gps() 
            
            # Convert for OpenCV processing
            bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

            # Thread-safe update of shared resources
            with self.lock:
                self.frame = bgr_frame
                self.gps_data = current_gps

    def get_latest(self):
        """Returns the most recent frame-GPS bundle."""
        with self.lock:
            return self.frame, self.gps_data

    def stop(self):
        self.running = False
        self.thread.join()
        self.picam2.stop()
