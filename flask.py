import cv2
from flask import Flask, Response, render_template
import numpy as np
import time
import threading
import os
import sys

# --- Configuration ---
NCNN_FOLDER = "ncnn"  # Path to your local ncnn folder
sys.path.insert(0, os.path.join(NCNN_FOLDER, "python"))  # Add ncnn python bindings to path

import ncnn  # Now imports from local folder

model_param = "yolo-opt.param"  # NCNN model files
model_bin = "yolo-opt.bin"
confidence_threshold = 0.3
frame_width = 320  # Reduced resolution for better performance
frame_height = 240
default_camera_index = 0

# --- Global Variables ---
app = Flask(__name__)
output_frame = None
lock = threading.Lock()
cap = None
net = None  # NCNN network
camera_is_running = False

# --- Load NCNN Model ---
def load_ncnn_model():
    global net
    try:
        net = ncnn.Net()
        # Enable Vulkan if available (faster on Pi 4)
        net.opt.use_vulkan_compute = True  
        net.load_param(model_param)
        net.load_model(model_bin)
        print("NCNN model successfully loaded from local folder")
        return True
    except Exception as e:
        print(f"Error loading NCNN model: {e}")
        return False

# --- Camera Initialization ---
def cam_initialize():
    global cap, camera_is_running
    if cap is not None:
        cap.release()

    for index in range(default_camera_index, 3):  # Try fewer indices
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            cap = temp_cap
            camera_is_running = True
            return True
        temp_cap.release()
    
    camera_is_running = False
    return False

# --- NCNN Detection ---
def ncnn_detect(frame):
    mat_in = ncnn.Mat.from_pixels(
        frame, 
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,  # Convert BGR to RGB
        frame.shape[1], 
        frame.shape[0]
    )
    
    # Normalize input (adjust based on your model)
    mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
    
    ex = net.create_extractor()
    ex.input("in0", mat_in)  # "in0" is common input name
    
    ret, mat_out = ex.extract("out0")  # "out0" is common output name
    
    # Parse detections (simplified - adjust for your model)
    detections = []
    if ret == 0:  # Success
        # Example parsing - this varies by model!
        for i in range(mat_out.h):
            values = mat_out.row(i)
            if values[4] > confidence_threshold:
                detections.append([
                    int(values[0] * frame.shape[1]),  # x1
                    int(values[1] * frame.shape[0]),  # y1
                    int(values[2] * frame.shape[1]),  # x2
                    int(values[3] * frame.shape[0]),  # y2
                    values[4],  # confidence
                    int(values[5])  # class_id
                ])
    return detections

# --- Draw Detections ---
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_id}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

# --- Frame Processing Thread ---
def process_frames():
    global output_frame, camera_is_running
    
    if not cam_initialize():
        print("Camera initialization failed")
        return

    print("Starting frame processing...")
    while camera_is_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        detections = []
        if net:
            try:
                detections = ncnn_detect(frame)
                frame = draw_detections(frame, detections)
            except Exception as e:
                print(f"Detection error: {e}")

        # Efficient JPEG encoding with quality tradeoff
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        
        with lock:
            output_frame = jpeg.tobytes()

        # Throttle CPU usage
        time.sleep(0.05)

    if cap:
        cap.release()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if output_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
                elif not camera_is_running:
                    break
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Main ---
if __name__ == '__main__':
    # Verify NCNN folder exists
    if not os.path.exists(NCNN_FOLDER):
        print(f"Error: NCNN folder not found at {NCNN_FOLDER}")
        print("Please clone NCNN with: git clone --recursive https://github.com/Tencent/ncnn.git")
        sys.exit(1)
    
    # Check if python bindings exist
    if not os.path.exists(os.path.join(NCNN_FOLDER, "python", "ncnn")):
        print("NCNN python bindings not found. Building them...")
        os.system(f"cd {NCNN_FOLDER} && mkdir -p build && cd build && cmake .. && make -j4")
    
    if load_ncnn_model():
        threading.Thread(target=process_frames, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
