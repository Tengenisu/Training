import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
obj_detection = ObjectDetection()

capture_video = cv2.VideoCapture('los_angeles.mp4')

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    return_frame, frame = capture_video.read()
    count += 1
    if not return_frame:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = obj_detection.detect(frame)
    for box in boxes:
        (x, y, width, height) = box
        cx = int((x + x + width) / 2)
        cy = int((y + y + height) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for point in center_points_cur_frame:
            for point2 in center_points_prev_frame:
                distance = math.hypot(point2[0] - point[0], point2[1] - point[1])

                if distance < 20:
                    tracking_objects[track_id] = point
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, point2 in tracking_objects_copy.items():
            object_exists = False
            for point in center_points_cur_frame_copy:
                distance = math.hypot(point2[0] - point[0], point2[1] - point[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = point
                    object_exists = True
                    if point in center_points_cur_frame:
                        center_points_cur_frame.remove(point)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for point in center_points_cur_frame:
            tracking_objects[track_id] = point
            track_id += 1

    for object_id, point in tracking_objects.items():
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (point[0], point[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)


    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(0)
    if key == 27:
        break

capture_video.release()
cv2.destroyAllWindows()
