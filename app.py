import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO
import time
import threading

model_path = 'yolo11n.pt'
#✦•┈๑⋅⋯ Camera Configurations ⋯⋅๑┈•✦
camera_index = 2
frame_width = 640
frame_height = 480
confidence_threshold = 0.3


#✦•┈๑⋅⋯Global Variables ⋯⋅๑┈•✦
app = Flask(__name__)
yolo_model = YOLO(model_path)
cap = None
output_frame = None
lock = threading.Lock()


def cam_initialize():
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Error initializing camera')
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    print(f'Camera {camera_index} successfully initialized')
    return True


def process_frames():
    global cap, output_frame, lock
    if not cam_initialize():
        print('Exiting due to camera initialization failure')
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to grab frame, retrying...')
                time.sleep(1)
                if not cam_initialize():
                    break
                continue
            results = yolo_model(frame, stream=False, verbose=False, conf=confidence_threshold)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
            # Encode frame for web stream
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            with lock:
                output_frame = jpeg.tobytes()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Frame processing stopped")


def generate_mjpeg_stream():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.1)
                continue
            frame_bytes = output_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def run_flask_app():
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    frame_thread = threading.Thread(target=process_frames)
    frame_thread.daemon = True
    frame_thread.start()
    run_flask_app()