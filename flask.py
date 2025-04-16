# yolo_stream.py
from ultralytics import YOLO
import cv2

model = YOLO("corha_ncnn_model")  # Load nano model for speed


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, imgsz=640, stream=True)
        for r in results:
            annotated_frame = r.plot()

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
