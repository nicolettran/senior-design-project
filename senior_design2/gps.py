import serial
import pynmea2
import threading

class GPSReader:
    """
    Handles asynchronous GPS data collection from the NEO 6M module.
    Prevents the 1Hz GPS update rate from slowing down the 30fps camera.
    """
    def __init__(self):
        # Default to /dev/serial0 for Raspberry Pi GPIO pins
        try:
            self.ser = serial.Serial("/dev/serial0", 9600, timeout=1)
        except Exception as e:
            print(f"GPS Serial Error: {e}")
            self.ser = None

        self.current_gps = {"lat": 0.0, "lng": 0.0, "alt": 0.0}
        self.running = True
        
        # Start background thread to keep current_gps updated
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running and self.ser:
            try:
                line = self.ser.readline().decode("ascii", errors="replace")
                if line.startswith("$GPGGA"):
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        self.current_gps = {
                            "lat": msg.latitude,
                            "lng": msg.longitude,
                            "alt": msg.altitude
                        }
            except Exception:
                continue

    def get_last_known_gps(self):
        """Returns the most recent coordinates without blocking."""
        return self.current_gps

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()

# Singleton instance to be shared across the project
gps_module = GPSReader()
