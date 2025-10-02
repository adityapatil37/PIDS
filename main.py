import os
import cv2
import time
import torch
import threading
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from pymongo import MongoClient
import datetime
from collections import defaultdict, deque
from numpy.linalg import norm

from flask import Flask, render_template, Response, jsonify 
from flask_socketio import SocketIO

# ------------------ CONFIG ------------------
YOLO_WEIGHTS = "yolov12n.pt"
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TILE_W, TILE_H = 480, 360
UNKNOWN_SAVE_DIR = "unknown_crops"
os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]
logs_col = db["logs"]

# Zones per camera
ZONES = {
    "Camera_1": "Entrance",
    "Camera_2": "Restricted Area",
    "Webcam": "Lobby"
}

print("Using device:", DEVICE)
yolo_model = YOLO(YOLO_WEIGHTS).to(DEVICE)
extractor = FeatureExtractor(model_name=REID_MODEL_NAME, model_path=REID_MODEL_PATH, device=DEVICE)

# Flask + SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

latest_frames = {}
frames_lock = threading.Lock()

# ------------------ HELPERS (same as your code) ------------------
def l2norm(v: np.ndarray):
    if v is None:
        return None
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / (n + 1e-12)

def load_known_people():
    people = []
    for doc in people_col.find():
        feats = []
        for f in doc.get("features", []):
            arr = np.array(f, dtype=np.float32).flatten()
            feats.append(l2norm(arr))
        if feats:
            people.append({"name": doc["name"], "role": doc.get("role", "Unknown"), "features": feats})
    return people

# ------------------ CAMERA PROCESSING ------------------
def process_camera(source, cam_name):
    print(f"[{cam_name}] starting, source={source}")
    cap = cv2.VideoCapture(source)
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)

    known_people = load_known_people()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo_model(frame, verbose=False)
        detections = []
        for r in results[0].boxes:
            cls_id, conf = int(r.cls[0]), float(r.conf[0])
            if cls_id == 0 and conf >= 0.45:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w < 30 or h < 60:
                    continue
                detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            tid = track.track_id
            display_name, role = "UNKNOWN", "N/A"

            # ---- RE-ID Matching ----
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                feat = extractor([crop_rgb])[0].cpu().numpy()
                feat = l2norm(feat)

                # simple nearest-neighbor
                best_name, best_role, best_dist = None, None, 1.0
                for person in known_people:
                    dists = [float(cosine(feat, ex)) for ex in person["features"]]
                    if dists and min(dists) < best_dist:
                        best_dist = min(dists)
                        best_name, best_role = person["name"], person["role"]

                if best_dist < 0.55:
                    display_name, role = best_name, best_role

            # ---- Annotate ----
            color = (0, 255, 0) if display_name != "UNKNOWN" else (0, 0, 255)
            label = f"{display_name} ({role})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # ---- Log to DB ----
            log = {
                "track_id": tid,
                "name": display_name,
                "role": role,
                "camera": cam_name,
                "zone": ZONES.get(cam_name, "Unknown"),
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            result = logs_col.insert_one(log)
            log["_id"] = str(result.inserted_id)   # convert ObjectId to string
            socketio.emit("person_detected", log)


        # Update latest frame
        with frames_lock:
            latest_frames[cam_name] = cv2.resize(frame, (TILE_W, TILE_H))

    cap.release()
    print(f"[{cam_name}] exiting")
    
    

def generate_frames(camera_url, camera_name):
    cap = cv2.VideoCapture(camera_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ------------------ FLASK ROUTES ------------------
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/history")
def history():
    records = list(logs_col.find().sort("time", -1).limit(50))
    for r in records:
        r["_id"] = str(r["_id"])
    return jsonify(records)

@app.route('/video/<camera_name>')
def video_feed(camera_name):
    # map camera name to URL/path
    cameras = {
        "Camera_1": "vid3.mp4",
        "Camera_2": "vid4.mp4"
    }
    return Response(
        generate_frames(cameras[camera_name], camera_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ------------------ BACKGROUND THREADS ------------------
def start_cameras(camera_list):
    for src, cam in camera_list:
        t = threading.Thread(target=process_camera, args=(src, cam), daemon=True)
        t.start()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    camera_list = [
        ("vid3.mp4", "Camera_1"),
        ("vid4.mp4", "Camera_2"),
        # (0, "Webcam"),
    ]
    start_cameras(camera_list)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
