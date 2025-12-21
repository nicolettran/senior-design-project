from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from datetime import datetime
import cv2
import serial
import threading
import pynmea2

app = Flask(__name__)

# GPS setup
gps_lat = "Waiting for GPS..."
gps_lng = ""

def gps_reader():
    global gps_lat, gps_lng
    port = "/dev/serial0"
    ser = serial.Serial(port, baudrate=9600, timeout=1)

    while True:
        try:
            data = ser.readline().decode("ascii", errors="replace").strip()
            if data.startswith("$GPRMC"):
                msg = pynmea2.parse(data)
                if msg.status == 'A':
                    gps_lat = f"Lat: {msg.latitude:.6f}"
                    gps_lng = f"Lng: {msg.longitude:.6f}"
                else:
                    gps_lat = "No GPS connected"
                    gps_lng = "Attempting to connect to satellites"
        except:
            pass

# Start background GPS thread
threading.Thread(target=gps_reader, daemon=True).start()


# Camera setup
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()


# GPS overlay box function
def draw_text_with_bg(frame, text, position, font, scale, color, thickness):
    x, y = position

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    # Create semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 8), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


# Streaming camera data to website
def generate_frames():
    while True:
        frame = camera.capture_array()

        # GPS text overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wsu_gold = (0, 205, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        draw_text_with_bg(frame, f"Time: {timestamp}", (15, 30), font, scale, wsu_gold, thickness)
        draw_text_with_bg(frame, gps_lat, (15, 60), font, scale, wsu_gold, thickness)
        draw_text_with_bg(frame, gps_lng, (15, 90), font, scale, wsu_gold, thickness)

        # Camera frames
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
# Website 
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Camera Stream</title>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: #f4f4f9;
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    header {
        width: 100%;
        background: #000000; /* Black header */
        padding: 12px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        flex-shrink: 0;
    }

    header h1 {
        font-size: clamp(0.9rem, 2vw, 1.3rem); /* Smaller, responsive font */
        font-weight: bold;
        margin: 0;
        letter-spacing: 0.5px;
        color: #FFCD00; /* WSU gold */
        white-space: nowrap; /* Keep text on one line */
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .container {
        flex: 1;
        display: flex;
        flex-direction: column;
        padding: 15px;
        overflow: hidden;
    }

    .video-wrapper {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        background: #ffffff; /* White card */
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 15px;
        overflow: hidden;
    }

    #video-stream {
        max-width: 100%;
        max-height: 100%;
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 6px;
        background: none;
        border: none;
    }

    footer {
        padding: 10px;
        font-size: 0.8rem;
        color: #555;
        text-align: center;
        background: white;
        flex-shrink: 0;
    }

    /* Mobile adjustments */
    @media (max-width: 768px) {
        .container { padding: 10px; }
        .video-wrapper { padding: 10px; }
        footer { font-size: 0.7rem; padding: 8px; }
    }
    @media (max-width: 480px) {
        header { padding: 10px 0; }
    }
</style>
</head>
<body>
    <header>
        <h1>Senior Design 1 - Open House | Camera Streaming System</h1>
    </header>
    
    <div class="container">
        <div class="video-wrapper">
            <img id="video-stream" src="{{ url_for('video_feed') }}" alt="Camera Stream">
        </div>
    </div>
    
    <footer>
        Wichita State University | Senior Design Project 2025
    </footer>
</body>
</html>
""")

# Video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
