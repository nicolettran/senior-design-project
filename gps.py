import serial
import time
import pynmea2

print("Starting GPS Reader...")

port = "/dev/serial0"
ser = serial.Serial(port, baudrate=9600, timeout=0.5)

try:
    while True:
        newdata = ser.readline().decode("ascii", errors="replace").strip()

        if newdata.startswith("$GPRMC"):
            newmsg = pynmea2.parse(newdata)
            lat = newmsg.latitude
            lng = newmsg.longitude
            gps = f"Latitude: {lat} Longitude: {lng}"
            print(gps)

except KeyboardInterrupt:
    print("\nGPS reader stopped by user.")

finally:
    ser.close()
    print("Serial port closed safely.")
