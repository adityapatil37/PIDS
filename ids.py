# app.py
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
import winsound
import tkinter as tk
from tkinter import messagebox

YOLO_WEIGHTS = "yolov12n.pt"
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Matching params (tune these)
STRICT_TH = 0.35   # strong match
LOOSE_TH = 0.55        # looser cross-camera match
RATIO_MARGIN = 0.75    # ratio test: best / second_best must be < RATIO_MARGIN
YOLO_CONF_TH = 0.45    # detection confidence threshold

# re-ID dynamic update params
REID_INTERVAL = 5            # check re-id every N frames per track
MIN_CONF_ASSIGN = 0.55      # if track unknown and conf >= this -> assign
CONF_MARGIN = 0.18          # new_conf must exceed old_conf by this to override immediately
CONSECUTIVE_UPDATES = 2     # require this many repeated confirmations to switch identity
EMA_ALPHA_TRACK = 0.85      # EMA for updating per-track stored feature


TILE_W, TILE_H = 480, 360

UNKNOWN_SAVE_DIR = "unknown_crops"
os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]
logs_col = db["logs"]
history_col = db["track_history"]

print("Using device:", DEVICE)
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to(DEVICE)

extractor = FeatureExtractor(
    model_name=REID_MODEL_NAME,
    model_path=REID_MODEL_PATH,
    device=DEVICE
)

ALERT_UNKNOWN_SECONDS = 5   # must stay unknown for >=3 sec
ALERT_COOLDOWN = 10         # minimum 10 sec between alerts per camera

last_alert_time = {}

def trigger_alert(frame, cam_name, tid):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ALERT] UNKNOWN detected on {cam_name} (track {tid}) at {ts}")

    # Save snapshot
    snap_path = f"alert_snapshots/{cam_name}_{tid}_{int(time.time())}.jpg"
    os.makedirs("alert_snapshots", exist_ok=True)
    cv2.imwrite(snap_path, frame)

    # Play sound (non-blocking)
    threading.Thread(target=lambda: winsound.Beep(1000, 700), daemon=True).start()

    # GUI popup
    show_popup(cam_name, tid)

# Track last log time per (camera, track_id) to avoid duplicate logs
last_log_times = {}
LOG_INTERVAL = 5


def log_person_event(name, cam_name, tid, frame, bbox):
    now = time.time()
    key = (cam_name, tid)
    if key in last_log_times and now - last_log_times[key] < LOG_INTERVAL:
        return
    last_log_times[key] = now

    ts = datetime.datetime.utcnow()

    # Crop only the detected person
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    person_crop = frame[y1:y2, x1:x2]

    # Ensure the folder exists
    thumb_dir = os.path.join("static", "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    # Generate filename
    thumb_name = f"{cam_name}_{tid}_{int(time.time())}.jpg"
    thumb_path = os.path.join(thumb_dir, thumb_name)

    # Save cropped image
    import cv2
    cv2.imwrite(thumb_path, person_crop)

    # Store web path for the UI
    web_thumb_path = f"/static/thumbnails/{thumb_name}"

    history_col.insert_one({
        "timestamp": ts,
        "person_name": name,
        "camera_name": cam_name,
        "track_id": tid,
        "thumbnail": web_thumb_path
    })
    print(f"[DB] Logged {name} on {cam_name}, track {tid} at {ts}")
    
def show_popup(cam_name, tid):
    def popup():
        root = tk.Tk()
        root.title("🚨 Security Alert!")
        root.geometry("300x150+100+100")
        root.attributes("-topmost", True)  # always on top

        msg = tk.Label(
            root,
            text=f"UNKNOWN person detected!\n\nCamera: {cam_name}\nTrack ID: {tid}",
            fg="red", font=("Arial", 12), justify="center"
        )
        msg.pack(expand=True, padx=10, pady=20)

        # Auto-close after 5 sec
        root.after(5000, root.destroy)
        root.mainloop()

    threading.Thread(target=popup, daemon=True).start()



def l2norm(v: np.ndarray):
    if v is None:
        return None
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / (n + 1e-12)

def extract_feature(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    feat = extractor([crop_rgb])[0].cpu().numpy()

    feat = feat / (norm(feat) + 1e-12)
    return feat

def load_known_people():
    """Load people from MongoDB; convert stored features to numpy arrays (l2-normalized)."""
    people = []
    for doc in people_col.find():
        feats = []
        for f in doc.get("features", []):
            arr = np.array(f, dtype=np.float32).flatten()
            feats.append(l2norm(arr))
        if len(feats) == 0:
            continue
        people.append({
            "name": doc["name"],
            "role": doc.get("role", "Unknown"),
            "features": feats
        })
    print(f"[DB] Loaded {len(people)} known people")
    return people

def match_person(feat: np.ndarray, known_people):
    """
    Return (name, role, best_dist, strong_bool).
    best_dist = min cosine distance to any exemplar across all people.
    strong_bool indicates a strong match by strict threshold + ratio test.
    """
    if feat is None or len(known_people) == 0:
        return None, None, None, False

    f = l2norm(feat)
    best_person = None
    best_dist = 1.0
    second_dist = 1.0

    for person in known_people:
        dists = [float(cosine(f, ex)) for ex in person["features"]]
        if not dists:
            continue
        min_d = min(dists)
        if min_d < best_dist:
            second_dist = best_dist
            best_dist = min_d
            best_person = person
        elif min_d < second_dist:
            second_dist = min_d

    if best_person is None:
        return None, None, None, False

    ratio_ok = (best_dist / (second_dist + 1e-12)) < RATIO_MARGIN

    strong = (best_dist < STRICT_TH) and ratio_ok
    weak = (best_dist < LOOSE_TH) and ratio_ok

    if strong:
        return best_person["name"], best_person["role"], best_dist, True
    elif weak:
        return best_person["name"] + " (?)", best_person["role"], best_dist, False
    else:
        return None, None, None, False


def dist_to_conf(dist):
    """
    Map distance to confidence [0..1].
    - dist <= STRICT_TH -> 1.0
    - dist >= LOOSE_TH -> 0.0
    - linearly interpolate in-between
    """
    if dist is None:
        return 0.0
    if dist <= STRICT_TH:
        return 1.0
    if dist >= LOOSE_TH:
        return 0.0
    # linear interpolation
    return float((LOOSE_TH - dist) / (LOOSE_TH - STRICT_TH))


def save_unknown_crop(frame, box, cam_name):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{cam_name}_{ts}.jpg"
    path = os.path.join(UNKNOWN_SAVE_DIR, fname)
    cv2.imwrite(path, crop)
    return path

def extract_feature_from_crop(frame, box, margin=0.05):
    # expand a bit to include context but not too much
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat_t = extractor([crop_rgb])
    feat = feat_t[0].cpu().numpy().flatten()
    return l2norm(feat)

latest_frames = {}
frames_lock = threading.Lock()

from collections import defaultdict, deque

def process_camera(source, cam_name, known_reload_interval=30):
    print(f"[{cam_name}] starting, source={source}")
    cap = cv2.VideoCapture(source)
    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)

    track_info = {}
    track_vote = defaultdict(lambda: deque(maxlen=CONSECUTIVE_UPDATES))

    known_people = load_known_people()
    last_known_load = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] stream ended")
            break
        frame_idx += 1

        if time.time() - last_known_load > known_reload_interval:
            known_people = load_known_people()
            last_known_load = time.time()

        try:
            results = yolo_model(frame, verbose=False)
        except Exception as e:
            print(f"[{cam_name}] YOLO error: {e}")
            results = None

        detections = []
        if results is not None:
            for r in results[0].boxes:
                cls_id, conf = int(r.cls[0]), float(r.conf[0])
                if cls_id == 0 and conf >= YOLO_CONF_TH:
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    if w < 30 or h < 60:
                        continue
                    detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            try:
                x1, y1, x2, y2 = [int(v) for v in ltrb]
            except Exception:
                l, t, w, h = [int(v) for v in ltrb]
                x1, y1, x2, y2 = l, t, l + w, t + h

            tid = track.track_id
            info = track_info.get(tid, {
                "name": "UNKNOWN",
                "role": "",
                "conf": 0.0,
                "last_feat": None,
                "last_reid_frame": -999,
                "last_seen": time.time()
            })
            info["last_seen"] = time.time()
                # --- ALERT CHECK ---
            if info["name"] == "UNKNOWN":
                elapsed = time.time() - info.get("first_unknown_time", time.time())
                info.setdefault("first_unknown_time", time.time())
                if elapsed >= ALERT_UNKNOWN_SECONDS:
                    if time.time() - last_alert_time.get(cam_name, 0) > ALERT_COOLDOWN:
                        trigger_alert(frame, cam_name, tid)
                        last_alert_time[cam_name] = time.time()
            else:
                info.pop("first_unknown_time", None)


            if (frame_idx - info["last_reid_frame"]) >= REID_INTERVAL:
                feat = extract_feature_from_crop(frame, (x1, y1, x2, y2))
                info["last_reid_frame"] = frame_idx

                if feat is not None:
                    name, role, dist, strong = match_person(feat, known_people)
                    new_conf = dist_to_conf(dist) if dist is not None else 0.0

                    # Case A: currently unknown
                    if info["name"] == "UNKNOWN":
                        if new_conf >= MIN_CONF_ASSIGN:

                            info["name"] = name if name else "UNKNOWN"
                            info["role"] = role if role else ""
                            info["conf"] = new_conf
                            info["last_feat"] = feat.copy()
                        else:
                            if name:
                                track_vote[tid].append(name)
                                if list(track_vote[tid]).count(name) >= CONSECUTIVE_UPDATES and new_conf > 0.25:
                                    info["name"] = name
                                    info["role"] = role
                                    info["conf"] = new_conf
                                    info["last_feat"] = feat.copy()
                    else:
                        # Case B: currently assigned some identity
                        current_name = info["name"]
                        current_conf = info["conf"]

                        if name == current_name:
                            info["conf"] = max(current_conf, new_conf)
                            if info["last_feat"] is None:
                                info["last_feat"] = feat.copy()
                            else:
                                info["last_feat"] = EMA_ALPHA_TRACK * info["last_feat"] + (1 - EMA_ALPHA_TRACK) * feat
                                info["last_feat"] = l2norm(info["last_feat"])
                        else:
                            if new_conf > current_conf + CONF_MARGIN and new_conf >= MIN_CONF_ASSIGN and strong:
                                info["name"] = name
                                info["role"] = role
                                info["conf"] = new_conf
                                info["last_feat"] = feat.copy()
                                track_vote[tid].clear()
                            else:
                                if name:
                                    track_vote[tid].append(name)
                                    counts = {c: track_vote[tid].count(c) for c in set(track_vote[tid])}
                                    top_candidate, top_count = max(counts.items(), key=lambda x: x[1])
                                    if top_candidate != current_name and top_count >= CONSECUTIVE_UPDATES and new_conf > 0.25:
                                        rp = next((p for p in known_people if p["name"] == top_candidate), None)
                                        info["name"] = top_candidate
                                        info["role"] = rp["role"] if rp else ""
                                        info["conf"] = new_conf
                                        info["last_feat"] = feat.copy()

            track_info[tid] = info

            display_name = info["name"]
            color = (0, 200, 0) if display_name != "UNKNOWN" else (0, 0, 255)
            label = f"{display_name}" + (f" ({info['role']})" if info.get("role") else "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            
            if info["name"] != "UNKNOWN":
                bbox = (x1, y1, x2 - x1, y2 - y1)  # width/height style bbox
                log_person_event(info["name"], cam_name, tid, frame, bbox)


        with frames_lock:
            latest_frames[cam_name] = cv2.resize(frame, (TILE_W, TILE_H))

    cap.release()
    print(f"[{cam_name}] exiting")

def draw_cam_label(frame, cam_name):
    # draw semi-transparent strip and put text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, cam_name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def dashboard_loop(camera_list):
    n = len(camera_list)
    if n == 0:
        return
    grid_cols = int(np.ceil(np.sqrt(n)))
    grid_rows = int(np.ceil(n / grid_cols))

    while True:
        with frames_lock:
            frames = []
            for _, cam_name in camera_list:
                f = latest_frames.get(cam_name)
                if f is None:
                    f = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                else:
                    f = f.copy()
                f = draw_cam_label(f, cam_name)
                frames.append(f)

        while len(frames) < grid_cols * grid_rows:
            frames.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

        rows = []
        for r in range(grid_rows):
            row_frames = frames[r * grid_cols:(r + 1) * grid_cols]
            row = np.hstack(row_frames)
            rows.append(row)
        grid = np.vstack(rows)

        cv2.imshow("Dashboard", grid)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_list = [
        # sources
        ("vid3.mp4", "Camera_1"),
        ("vid4.mp4", "Camera_2"),
        # (0, "Webcam") 
    ]

    threads = []
    for src, cam in camera_list:
        t = threading.Thread(target=process_camera, args=(src, cam), daemon=True)
        t.start()
        threads.append(t)

    try:
        dashboard_loop(camera_list)
    except KeyboardInterrupt:
        print("Interrupted by user")

    for t in threads:
        t.join(timeout=1.0)
    print("Exiting.")