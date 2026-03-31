import ncnn
import cv2
import numpy as np

# Path to your exported NCNN YOLOv11 model
MODEL_PARAM = "vegetation_model.param"
MODEL_BIN   = "vegetation_model.bin"

# Load NCNN network
net = ncnn.Net()
net.load_param(MODEL_PARAM)
net.load_model(MODEL_BIN)

# Detection parameters
CONF_THRESHOLD = 0.5
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

def detect_vegetation(frame):
    # Runs NCNN YOLO inference on a frame and returns vegetation bounding boxes.
    

    orig_h, orig_w = frame.shape[:2]

    # Resize and normalize frame
    blob = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = blob[:, :, ::-1]  # BGR -> RGB
    blob = blob.astype(np.float32) / 255.0  # normalize
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW

    # Create NCNN Mat
    mat_in = ncnn.Mat(blob)

    # Run inference
    ex = net.create_extractor()
    ex.set_light_mode(True)
    ex.set_num_threads(4)
    ex.input("input.1", mat_in)

    ret, mat_out = ex.extract("output")  # 'output' is typical for exported YOLOv11 NCNN

    boxes = []

    if ret:
        # mat_out: [num_detections, 6] = [x1, y1, x2, y2, confidence, class_id]
        for i in range(mat_out.h):
            data = mat_out.row(i)
            conf = data[4]
            if conf > CONF_THRESHOLD:
                # Scale boxes back to original frame size
                x1 = int(data[0] * orig_w / INPUT_WIDTH)
                y1 = int(data[1] * orig_h / INPUT_HEIGHT)
                x2 = int(data[2] * orig_w / INPUT_WIDTH)
                y2 = int(data[3] * orig_h / INPUT_HEIGHT)
                boxes.append((x1, y1, x2, y2))

    return boxes