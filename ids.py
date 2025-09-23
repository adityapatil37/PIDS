import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from pymongo import MongoClient
import datetime
import threading
from collections import defaultdict, deque

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]
logs_col = db["logs"]

# Load known people
def load_known_people():
    people = []
    for doc in people_col.find():
        people.append({
            "name": doc["name"],
            "role": doc.get("role", "Unknown"),
            "features": np.array(doc["features"])
        })
    return people

# Torchreid extractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# YOLO model
yolo_model = YOLO("yolov12n.pt")

# Thresholds
STRICT_TH = 0.30
LOOSE_TH = 0.45
YOLO_CONF_TH = 0.6

# History buffer for smoothing
prediction_history = defaultdict(lambda: deque(maxlen=5))

# Shared latest frames
latest_frames = {}
lock = threading.Lock()

def extract_feature(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    feat = extractor([crop_rgb])[0].cpu().numpy()
    return feat

def match_person(feature, known_people):
    best_match = None
    best_dist = 1.0
    for person in known_people:
        dists = [cosine(feature, np.array(f).flatten()) for f in person["features"]]
        min_dist = min(dists)
        if min_dist < best_dist:
            best_dist = min_dist
            best_match = person

    if best_match and best_dist < STRICT_TH:
        return best_match["name"], best_match["role"], best_dist
    elif best_match and best_dist < LOOSE_TH:
        return best_match["name"] + " (?)", best_match["role"], best_dist
    else:
        return None, None, None

def process_camera(source, cam_name):
    cap = cv2.VideoCapture(source)
    known_people = load_known_people()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        for r in results[0].boxes:
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])

            if cls_id == 0 and conf > YOLO_CONF_TH:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                feature = extract_feature(frame, (x1, y1, x2, y2))
                if feature is None:
                    continue

                name, role, dist = match_person(feature, known_people)

                box_id = f"{cam_name}_{x1}_{y1}_{x2}_{y2}"
                if name:
                    prediction_history[box_id].append(name)
                else:
                    prediction_history[box_id].append("UNKNOWN")

                stable_name = max(set(prediction_history[box_id]),
                                  key=prediction_history[box_id].count)

                if stable_name != "UNKNOWN":
                    label = f"{stable_name} ({role})"
                    color = (0, 255, 0)
                else:
                    label = "UNKNOWN"
                    color = (0, 0, 255)
                    logs_col.insert_one({
                        "timestamp": datetime.datetime.utcnow(),
                        "camera": cam_name,
                        "features": feature.tolist(),
                        "status": "unknown"
                    })

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Store latest frame safely
        with lock:
            latest_frames[cam_name] = cv2.resize(frame, (480, 360))

    cap.release()

def dashboard_loop(camera_list):
    while True:
        with lock:
            frames = [latest_frames.get(cam_name, np.zeros((360, 480, 3), dtype=np.uint8))
                      for _, cam_name in camera_list]

        # Arrange frames in grid
        rows = []
        row_size = int(np.ceil(np.sqrt(len(frames))))
        for i in range(0, len(frames), row_size):
            row = np.hstack(frames[i:i+row_size])
            rows.append(row)
        grid = np.vstack(rows)

        cv2.imshow("Dashboard", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_list = [
        ("vid3.mp4", "Camera_1"),
        ("vid4.mp4", "Camera_2"),
        # Add more here
    ]

    threads = []
    for path, cam_name in camera_list:
        t = threading.Thread(target=process_camera, args=(path, cam_name))
        t.start()
        threads.append(t)

    # Run dashboard in main thread
    dashboard_loop(camera_list)

    for t in threads:
        t.join()
