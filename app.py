import cv2
import threading
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine

# ----------------- Models -----------------

# YOLOv11 or v12
yolo_model = YOLO("yolov12n.pt")

# Torchreid feature extractor (light model for speed)
extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x0_25_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ----------------- Global ReID State -----------------

global_gallery = {}   # {global_id: feature_vector}
next_global_id = 0
lock = threading.Lock()


# ----------------- Helper Functions -----------------

def extract_reid_feature(frame, box):
    """Extract a ReID feature vector from a bounding box region."""
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    feat = extractor([crop_rgb])
    return feat[0].cpu().numpy()


def match_global_id(feature, threshold=0.5):
    """Match feature against global gallery, return a global ID."""
    global next_global_id
    with lock:
        for gid, g_feat in global_gallery.items():
            if cosine(feature, g_feat) < threshold:
                # Update gallery with running average for stability
                global_gallery[gid] = (g_feat + feature) / 2
                return gid
        # New person
        gid = next_global_id
        global_gallery[gid] = feature
        next_global_id += 1
        return gid


# ----------------- Camera Thread -----------------

def process_camera(cam_url, cam_name):
    """Run YOLO + DeepSORT + ReID on a single camera stream."""

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
            if cls_id == 0:  # person
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        # Update per-camera DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            # Local track box
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]

            # Extract ReID feature
            feature = extract_reid_feature(frame, (x1, y1, x2, y2))
            if feature is None:
                continue

            # Match across cameras (global ID assignment)
            global_id = match_global_id(feature)

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"GID {global_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(cam_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(cam_name)


# ----------------- Main -----------------

if __name__ == "__main__":
    camera_streams = [  # webcam
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
