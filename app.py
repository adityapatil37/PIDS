import cv2
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv11/12 model (choose yolov11n.pt, yolov12n.pt, etc.)
yolo_model = YOLO("yolov12n.pt")

# Shared global ID map across cameras
global_id_map = {}
next_global_id = 0
lock = threading.Lock()


def process_camera(cam_url, cam_name):
    """Run YOLO + DeepSORT on a single camera stream in its own thread"""

    global next_global_id

    # Each camera gets its own DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(cam_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = yolo_model(frame, verbose=False)

        detections = []
        for r in results[0].boxes:
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])
            if cls_id == 0:  # person class only
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        # Update this camera's DeepSORT tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            local_id = track.track_id
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]

            with lock:
                if (cam_name, local_id) not in global_id_map:
                    global_id_map[(cam_name, local_id)] = next_global_id
                    next_global_id += 1
                global_id = global_id_map[(cam_name, local_id)]

            # Draw box + global ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"GID {global_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(cam_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(cam_name)


if __name__ == "__main__":
    # Example sources: webcam + RTSP/IP cams
    camera_streams = [
        "vid1.mp4",
        "vid2.mp4"
    ]

    threads = []
    for i, cam_url in enumerate(camera_streams):
        t = threading.Thread(target=process_camera, args=(cam_url, f"Camera {i}"), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    cv2.destroyAllWindows()
