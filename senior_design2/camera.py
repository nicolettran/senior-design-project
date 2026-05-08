# Real-time Camera Stream with GPS Synchronization
# PURPOSE:
#   Captures video frames from the Raspberry Pi camera at ~30 FPS and
#   "bundles" each frame with the latest GPS coordinates from gps.py.
#   Threading ensures the slow 1 Hz GPS update rate never blocks the
#   fast 30 FPS camera loop — both run independently and share data safely.
#
# HOW IT FITS INTO THE SYSTEM:
#   main.py calls stream.get_latest() every loop iteration to receive
#   the most recent (frame, gps_data) pair. That pair is then passed
#   into the ML vegetation detector, the CV powerline detector, and
#   ultimately the risk evaluator and logger.
# ============================================================================

from picamera2 import Picamera2   # Raspberry Pi official camera library
import cv2                        # OpenCV — used here for color space conversion
import threading                  # Python standard library for background thread
from gps import gps_module        # Shared GPS singleton defined in gps.py


class CameraStream:
    """
    Maintains a high-speed camera thread that 'bundles' GPS data with frames.
    This ensures that the image and the coordinates are synced in time.

    Why a background thread?
        The camera produces frames at ~30 FPS. GPS updates arrive at only 1 Hz.
        If we waited for GPS on every frame we'd either slow the camera down
        or miss frames entirely. Instead, this thread grabs the most recent
        GPS reading that the GPS thread already has cached in memory — zero
        blocking, always in sync.
    """

    def __init__(self):
        # Camera Setup
        self.picam2 = Picamera2()

        # Configure for video capture at 640x480 - balanced resolution for
        # real-time ML inference on the Raspberry Pi's limited CPU and camera size
        config = self.picam2.create_video_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()

        # AfMode 2 = Continuous autofocus to keep vegetation sharp as the
        # drone moves at varying distances from the power lines below
        self.picam2.set_controls({"AfMode": 2})

        # Shared state (accessed by both background and main thread)
        self.frame    = None   # Most recent BGR frame from the camera
        self.gps_data = None   # GPS coordinates paired with that frame

        # Thread control flag allows clean shutdown on KeyboardInterrupt
        self.running = True    # Set to False by stop() to end the loop cleanly

        # Lock ensures frame and gps_data are always updated together
        # prevents main.py from reading a new frame paired with stale GPS data
        self.lock = threading.Lock()

        # daemon=True means this thread dies automatically when the main program
        # exits, so the drone system never hangs waiting for this thread on shutdown
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """
        Background loop runs continuously at camera speed (~30 FPS).
        Each iteration captures one frame and pairs it with the latest GPS fix.
        """
        while self.running:
            # Pull the newest frame directly from the camera's hardware buffer
            raw_frame = self.picam2.capture_array()

            # Ask the GPS module for the most recently parsed coordinate.
            # This is a non-blocking memory read, NOT a serial port read,
            # so it never introduces delay into this fast loop
            current_gps = gps_module.get_last_known_gps()

            # Picamera2 returns RGB arrays; OpenCV expects BGR, convert so
            # all downstream processing (detection, drawing, saving) works correctly
            bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

            # Acquire the lock before writing so main.py never reads a
            # half-updated frame/gps pair mid-assignment
            with self.lock:
                self.frame    = bgr_frame
                self.gps_data = current_gps

    def get_latest(self):
        """
        Called by main.py on every loop iteration.
        Returns the most recent (frame, gps_data) bundle as an atomic pair.
        The lock guarantees the frame and GPS always belong to the same moment.
        """
        with self.lock:
            return self.frame, self.gps_data

    def stop(self):
        """
        Cleanly shuts down the camera system on KeyboardInterrupt or program exit.
        Signals the background thread to stop, waits for it to finish, then
        releases the camera hardware so it isn't left in a locked state.
        """
        self.running = False
        self.thread.join()   # Block until the background thread fully exits
        self.picam2.stop()   # Release the physical camera hardware resource
