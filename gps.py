def get_gps():
    # Placeholder, will get code from Olivia
    return {
        "lat": 37.715921,
        "lng": -97.286581,
        "alt": -97.286581
    }

"""
import serial
import pynmea2

# Example for Raspberry Pi: /dev/serial0 or /dev/ttyAMA0
ser = serial.Serial("/dev/serial0", 9600, timeout=1)

def get_gps():
    while True:
        line = ser.readline().decode("ascii", errors="replace")
        if line.startswith("$GPGGA"):
            msg = pynmea2.parse(line)
            if msg.lat and msg.lon:
                lat = msg.latitude
                lon = msg.longitude
                alt = msg.altitude
                return {"lat": lat, "lng": lon, "alt" : alt}
"""