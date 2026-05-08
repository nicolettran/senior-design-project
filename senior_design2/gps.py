# Asynchronous GPS Data Reader with NEO-6M Module
# PURPOSE:
#   Continuously reads NMEA sentences from the NEO-6M GPS module over the
#   Raspberry Pi's hardware serial port and caches the latest coordinates.
#   Other modules (camera.py) call get_last_known_gps() to retrieve those
#   cached coordinates instantly without ever touching the serial port themselves.
#
# WHY ASYNC?
#   The GPS module outputs data at only 1 Hz (one fix per second), while the
#   camera runs at 30 FPS. If the camera waited for a fresh GPS read each frame,
#   it would be bottlenecked to 1 FPS. This module decouples the two rates:
#   GPS updates in the background, camera reads from memory at full speed.
#
# HOW IT FITS INTO THE SYSTEM:
#   gps.py creates a single shared instance (gps_module) at the bottom of
#   this file. camera.py imports that instance and calls get_last_known_gps()
#   inside its own background thread to geotag every captured frame.
# ============================================================================

import serial    # pyserial communicates with the GPS over UART serial port
import pynmea2   # Parses raw NMEA sentence strings into structured GPS objects
import threading # Python standard library for running the read loop in background


class GPSReader:
    """
    Handles asynchronous GPS data collection from the NEO-6M module.
    Prevents the 1 Hz GPS update rate from slowing down the 30 FPS camera.
    """

    def __init__(self):
        # Serial Port Setup
        # /dev/serial0 is the default hardware UART on Raspberry Pi GPIO pins
        # 9600 baud is the NEO-6M's default communication speed
        # timeout=1 prevents readline() from blocking forever if no data arrives
        try:
            self.ser = serial.Serial("/dev/serial0", 9600, timeout=1)
        except Exception as e:
            # GPS hardware may not be connected during bench testing, instead
            # of crashing the program we catch the exception, set ser to None,
            # and rely on get_last_known_gps() to return fallback coordinates until
            # a real fix is acquired, this allows development to proceed without GPS hardware
            print(f"GPS Serial Error: {e}")
            self.ser = None

        # Cache that stores the latest valid GPS fix.
        # Initialized to 0.0 so get_last_known_gps() can detect "no fix yet"
        self.current_gps = {"lat": 0.0, "lng": 0.0, "alt": 0.0}

        self.running = True

        # Start the background reader thread so GPS updates happen continuously
        # daemon=True ensures this thread shuts down with the main program
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """
        Background loop — reads one NMEA sentence per iteration from the serial port.
        Runs at the GPS module's natural 1 Hz update rate.
        """
        while self.running and self.ser:
            try:
                # readline() blocks until a full newline-terminated sentence arrives
                # decode as ASCII since NMEA is plain-text; replace bad bytes safely
                line = self.ser.readline().decode("ascii", errors="replace")

                # NMEA outputs many sentence types ($GPRMC, $GPGSV, etc.)
                # $GPGGA (Global Positioning Fix Data) contains key GPS info: lat, lng, alt
                if line.startswith("$GPGGA"):
                    msg = pynmea2.parse(line)   # Parse raw string into a structured object

                    # Only update the cache if we have a real fix (non-zero coordinates).
                    # A fix of 0.0/0.0 means the module hasn't acquired satellites yet
                    if msg.latitude and msg.longitude:
                        self.current_gps = {
                            "lat": msg.latitude,
                            "lng": msg.longitude,
                            "alt": msg.altitude
                        }
            except Exception:
                # Malformed NMEA sentences or serial glitches are common outdoors
                # just silently continue rather than crashing the background thread
                continue

    def get_last_known_gps(self):
        """
        Returns the most recent valid GPS fix, or fallback test coordinates
        if the module hasn't acquired a satellite lock yet.

        Called by camera.py's background thread on every frame capture.
        This is a fast in-memory read, no serial I/O happens here.
        """
        # lat == 0.0 signals that no real fix has been received yet
        if self.current_gps["lat"] == 0.0:
            # Return a known test location (Wichita, KS in WSU area) so development
            # and bench testing can proceed without live GPS hardware outdoors
            return {
                "lat": 37.715921,
                "lng": -97.286581,
                "alt": 400.0
            }
        return self.current_gps

    def stop(self):
        """
        Cleanly shuts down the GPS reader on program exit.
        Stops the read loop and closes the serial port to free the hardware resource.
        """
        self.running = False
        if self.ser:
            self.ser.close()


# All other modules like camera.py import this instance rather than creating
# their own, ensuring every part of the system shares the same GPS data.
gps_module = GPSReader()
