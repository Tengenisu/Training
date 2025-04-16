import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO
import time
import threading
import os # Added for checking template folder

# --- Configuration ---
model_path = 'yolov11n.pt' # Make sure this model exists or change to a valid one e.g., yolov8n.pt
confidence_threshold = 0.3
frame_width = 640
frame_height = 480
default_camera_index = 0 # Start trying from this index

# --- Global Variables ---
app = Flask(__name__)
output_frame = None
lock = threading.Lock()
cap = None # Initialize cap globally as None
yolo_model = None # Initialize model globally as None
camera_is_running = False # Flag to indicate camera status

# --- Load Model ---
try:
    yolo_model = YOLO(model_path)
    print(f"Successfully loaded YOLO model from {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Decide if you want the app to exit or continue without detection
    # exit() # Uncomment to exit if model loading fails

# --- Camera Handling ---
def cam_initialize():
    global cap, camera_is_running
    print("Attempting to initialize camera...")
    if cap is not None:
        cap.release()
        print("Released previous camera instance.")

    # Try indices starting from default_camera_index up to 10
    for index in range(default_camera_index, 10):
        print(f"Trying device index {index}...")
        # Try different backends if default doesn't work (optional)
        # Example for Windows: cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        # Example for Linux: cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            # Verify settings applied (optional)
            actual_width = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f'Camera index {index} successfully initialized.')
            print(f'Resolution set: {frame_width}x{frame_height}, Actual: {int(actual_width)}x{int(actual_height)}')
            cap = temp_cap # Assign to global cap only on success
            camera_is_running = True
            return True
        else:
            print(f"Failed to open index {index}.")
            temp_cap.release() # Release the temporary handle

    print('Could not initialize any camera device.')
    cap = None # Ensure cap is None if all attempts failed
    camera_is_running = False
    return False

# --- Frame Processing Thread ---
def process_frames():
    global cap, output_frame, lock, camera_is_running, yolo_model

    if not cam_initialize():
        print('Exiting frame processing thread due to camera initialization failure.')
        return # Stop the thread if camera fails initially

    if yolo_model is None:
        print("YOLO model not loaded. Running camera feed without detection.")

    print("Starting frame processing loop...")
    while camera_is_running:
        if cap is None: # Double check if cap became None somehow
            print("Camera is not available. Attempting re-initialization...")
            if not cam_initialize():
                print("Re-initialization failed. Stopping frame processing.")
                break # Exit loop if re-init fails
            else:
                continue # Continue loop if re-init succeeds

        ret, frame = cap.read()

        if not ret:
            print('Failed to grab frame. Check camera connection.')
            # Optional: Attempt re-initialization after a delay
            time.sleep(2)
            if cap is not None:
                cap.release()
                cap = None
            if not cam_initialize():
                 print("Re-initialization failed after frame grab failure. Stopping thread.")
                 camera_is_running = False # Signal thread stop
                 break
            continue # Try reading again in the next iteration

        annotated_frame = frame # Default to original frame

        # Perform detection only if model loaded successfully
        if yolo_model:
            try:
                results = yolo_model(frame, stream=False, verbose=False, conf=confidence_threshold)
                # Check if results contain detections
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    annotated_frame = results[0].plot() # Use YOLO's plotting function
                # else: # No detections, keep original frame (already assigned)
                #    pass
            except Exception as e:
                 print(f"Error during YOLO prediction: {e}")
                 # Keep using the original frame if prediction fails

        # Encode frame for web stream
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("Failed to encode frame to JPEG.")
            continue # Skip this frame

        # Update the output frame under lock
        with lock:
            output_frame = jpeg.tobytes()

        # Small delay to prevent CPU hogging (optional, adjust as needed)
        # time.sleep(0.01)

    # Cleanup when the loop exits
    print("Frame processing loop stopped.")
    if cap is not None:
        cap.release()
        print("Camera released.")
    cv2.destroyAllWindows() # May not be effective in server environment but doesn't hurt
    with lock:
        output_frame = None # Clear the frame on exit


# --- MJPEG Stream Generator ---
def generate_mjpeg_stream():
    global output_frame, lock, camera_is_running
    print("Client connected to MJPEG stream.")
    while True:
        frame_bytes = None
        with lock:
            if output_frame is not None:
                frame_bytes = output_frame
            elif not camera_is_running:
                 # Optional: Send a placeholder image or break if camera stopped
                 print("Camera not running, stopping stream for this client.")
                 break # Exit generator if camera stopped

        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Wait briefly if there's no frame yet but camera is supposed to be running
            time.sleep(0.1)

    print("MJPEG stream generator finished for a client.")


# --- Flask Routes ---
@app.route('/')
def index():
    """Video streaming home page."""
    # Check if template exists
    template_path = os.path.join(app.template_folder, 'index.html')
    if not os.path.exists(template_path):
        return "Error: 'templates/index.html' not found."
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if not camera_is_running and output_frame is None:
         return "Camera not available or initialization failed.", 503 # Service Unavailable
    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Start Background Thread ---
# This code runs when the module is imported by Flask
print("Initializing application and starting background thread...")
frame_processor_thread = threading.Thread(target=process_frames)
frame_processor_thread.daemon = True # Allows app to exit even if thread is running
frame_processor_thread.start()
print("Background frame processing thread started.")


# --- Main Execution Block (for running python your_script_name.py) ---
if __name__ == '__main__':
    print("Running Flask app directly using app.run()...")
    # Note: `threaded=True` is default and often needed for concurrent requests.
    # `debug=False` and `use_reloader=False` are recommended when running directly
    # if you started the background thread globally, to avoid duplicate threads.
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
