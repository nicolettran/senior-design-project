# On-Device ML Vegetation Detection (YOLOv11 + NCNN)
# PURPOSE:
#   Runs a custom-trained YOLOv11 object detection model on each camera frame
#   to locate patches of vegetation (trees, shrubs, overgrowth) that could
#   be encroaching on distribution lines. Returns bounding boxes in original
#   image pixel coordinates for use by risk.py and logger.py.
#
# WHY NCNN?
#   NCNN is a neural network inference framework designed specifically for
#   mobile and embedded ARM processors (exactly what the Raspberry Pi uses).
#   It is significantly faster than running PyTorch or TensorFlow directly on
#   the Pi, making real-time 30 FPS inference feasible without a GPU.
#
# MODEL FORMAT:
#   The YOLOv11 model was trained externally a Google Colab Notebook,
#   then exported to NCNN format (.param architecture + .bin weights).
#   These two files are loaded once at module import and reused every frame.
#
# PIPELINE OVERVIEW:
#   Frame → Letterbox Resize → Normalize → NCNN Inference →
#   Confidence Filter → Undo Letterbox → Size Filter → NMS → Boxes
#
# HOW IT FITS INTO THE SYSTEM:
#   main.py calls detect_vegetation(frame) each loop iteration.
#   The returned list of (x1, y1, x2, y2) boxes goes to risk.py for
#   proximity evaluation and logger.py for drawing green overlays.
# ============================================================================

import ncnn    # NCNN Python bindings, lightweight ARM-optimized inference engine
import cv2     # OpenCV, used for resizing, padding, and NMS
import numpy as np


# Model File Paths
# .param : text file describing the network architecture (layers, connections)
# .bin   : binary file containing all trained weight values
MODEL_PARAM = "model-1_ncnn_model/model.ncnn.param"
MODEL_BIN   = "model-1_ncnn_model/model.ncnn.bin"

# Detection Thresholds
# Only keep detections where the model's confidence score exceeds this value.
# 0.25 is a standard starting point, low enough to catch partially hidden
# vegetation, high enough to reduce false positives from sky/ground texture
CONF_THRESHOLD = 0.25

# Non-Maximum Suppression threshold: if two boxes overlap by more than 45%
# (IoU > 0.45) they are considered duplicates and only the higher-confidence
# box is kept. Prevents the same bush from generating 5 overlapping boxes.
NMS_THRESHOLD = 0.45

# YOLOv11 models expect a square input of this size.
# 640×640 is the standard YOLOv11 input resolution, balancing detail with speed
INPUT_SIZE = 640

# Model Loading (runs once at import time)
net = ncnn.Net()

# Limit to 4 CPU threads, balances inference speed against leaving CPU headroom
# for the camera thread, GPS thread, and OS processes running simultaneously
net.opt.num_threads = 4

net.load_param(MODEL_PARAM)   # Load the network architecture
net.load_model(MODEL_BIN)     # Load the trained weights into the architecture


def detect_vegetation(frame):
    """
    Runs the YOLOv11 model on a single BGR frame and returns vegetation boxes.

    Parameters:
        frame (np.ndarray): BGR image from camera.py (640x480 expected)

    Returns:
        List of (x1, y1, x2, y2) tuples in ORIGINAL image pixel coordinates.
        Empty list if no vegetation detected or model returns an error.
    """

    orig_h, orig_w = frame.shape[:2]   # Original frame dimensions

    # Step 1: Letterbox Resize, fit the image into a 640×640 square 
    # while preserving aspect ratio
    # YOLOv11 requires a square 640×640 input, but our frame is 640×480 (4:3).
    # Naive stretching would distort objects and hurt detection accuracy.
    # Letterboxing preserves aspect ratio by:
    #   1. Scaling the image so its LONGEST side fits in 640px
    #   2. Padding the SHORTER side with gray (114) pixels to reach 640×640
    # We record the scale factor and padding so we can reverse them later.
    scale = INPUT_SIZE / max(orig_h, orig_w)   # Scale factor to fit longest side
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_w = (INPUT_SIZE - new_w) // 2          # Horizontal padding in pixels
    pad_h = (INPUT_SIZE - new_h) // 2          # Vertical padding in pixels

    resized = cv2.resize(frame, (new_w, new_h))

    # Fill a 640×640 canvas with gray (114 is the standard YOLOv8 pad value)
    blob = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.float32)

    # Place the resized image centered in the canvas
    blob[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # Step 2: Preprocessing — Normalize and Reformat
    # [:, :, ::-1]        : flip channels BGR → RGB (model was trained on RGB)
    # / 255.0             : scale pixel values from [0, 255] to [0.0, 1.0]
    # .transpose(2, 0, 1) : reorder axes from HWC (height, width, channel)
    #                       to CHW (channel, height, width) — NCNN's expected format
    blob = blob[:, :, ::-1] / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1))   # HWC → CHW

    # Wrap the numpy array in an NCNN Mat object for the inference engine
    mat_in = ncnn.Mat(blob)

    # Step 3: NCNN Inference
    # Create a fresh extractor for this frame (stateless per-frame inference)
    ex = net.create_extractor()

    # Light mode reduces memory usage by freeing intermediate layer buffers,
    # important on the Raspberry Pi's limited RAM
    ex.set_light_mode(True)

    # Feed the preprocessed image into the model's input node "in0"
    ex.input("in0", mat_in)

    # Run inference and extract results from output node "out0"
    # ret=0 means success; non-zero means the model failed to run
    ret, mat_out = ex.extract("out0")
    if ret != 0:
        return []   # Inference failed, return no detections rather than crashing

    # Step 4: Parse Raw Output
    # YOLOv11 output shape: (5, 8400)
    #   Rows (5)    : [center_x, center_y, width, height, confidence_score]
    #   Columns (8400): one "anchor" candidate box per column
    # 8400 = 3 detection scales × multiple grid positions covering the 640×640 input
    out = np.array(mat_out)   # shape: (5, 8400)

    cv_boxes  = []   # Boxes in OpenCV [x, y, w, h] format for NMS
    conf_list = []   # Confidence score for each box

    for i in range(out.shape[1]):
        score = float(out[4, i])

        # Skip low-confidence anchors, the vast majority of the 8400 candidates
        if score < CONF_THRESHOLD:
            continue

        # Extract box center and size in the 640×640 letterboxed space
        cx, cy, w, h = out[0, i], out[1, i], out[2, i], out[3, i]

        # Step 5: Undo Letterbox — Map Back to Original Coordinates
        # The box coordinates are in the padded 640×640 input space.
        # We reverse the padding offset and scale factor to get pixel positions
        # in the original 640×480 frame that the rest of the system uses.
        x1 = int((cx - w / 2 - pad_w) / scale)   # Left edge in original image
        y1 = int((cy - h / 2 - pad_h) / scale)   # Top edge in original image
        bw = int(w / scale)                        # Width in original image
        bh = int(h / scale)                        # Height in original image

        # Clamp to frame boundaries, boxes near the letterbox edge can go negative
        x1 = max(0, x1)
        y1 = max(0, y1)
        bw = min(bw, orig_w - x1)
        bh = min(bh, orig_h - y1)

        # Discard extremely small boxes (< 20×20 px) — too small to be meaningful
        # vegetation at the drone's operating altitude
        if bw < 20 or bh < 20:
            continue

        cv_boxes.append([x1, y1, bw, bh])   # OpenCV NMS format: [x, y, width, height]
        conf_list.append(score)

    if not cv_boxes:
        return []   # All candidates filtered out — no vegetation in this frame

    # Step 6: Non-Maximum Suppression (NMS)
    # NMS eliminates duplicate overlapping boxes for the same vegetation patch.
    # It keeps only the highest-confidence box when boxes overlap by > NMS_THRESHOLD.
    # Without NMS, one tree might generate 10–20 overlapping boxes.
    indices = cv2.dnn.NMSBoxes(cv_boxes, conf_list, CONF_THRESHOLD, NMS_THRESHOLD)

    # Step 7: Convert to (x1, y1, x2, y2) Format
    # OpenCV NMS returns indices into cv_boxes. We look up each kept box
    # and convert from [x, y, width, height] to corner format [x1, y1, x2, y2]
    # which is what risk.py and logger.py expect.
    boxes = []
    for idx in indices:
        # OpenCV version differences: idx may be a scalar int or a nested array
        i = int(idx[0]) if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
        x, y, w, h = cv_boxes[i]
        boxes.append((x, y, x + w, y + h))   # Convert to corner coordinates

    return boxes
