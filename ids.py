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

client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]
logs_col = db["logs"]

def load_known_people():
    people = []
    for doc in people_col.find():
        people.append({
            "name": doc["name"],
            "role": doc.get("role", "Unknown"),
            "features": np.array(doc["features"])
        })
    return people

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

yolo_model = YOLO("yolov12n.pt")

STRICT_TH = 0.30
LOOSE_TH = 0.45
YOLO_CONF_TH = 0.6

# History buffer for smoothing and can reducce it till 5 and so 
prediction_history = defaultdict(lambda: deque(maxlen=15))


latest_frames = {}
lock = threading.Lock()

def extract_feature(frame, box, margin=0.1):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(margin * w))
    y1 = max(0, y1 - int(margin * h))
    x2 = min(frame.shape[1], x2 + int(margin * w))
    y2 = min(frame.shape[0], y2 + int(margin * h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    feat = extractor([crop_rgb])[0].cpu().numpy()
    return feat / np.linalg.norm(feat)  


def match_person(feature, known_people):
    best_match, best_dist, second_best = None, 1.0, 1.0
    for person in known_people:
        person_feats = np.array(person["features"])
        dists = [cosine(feature, f.flatten()) for f in person_feats]
        min_dist = min(dists)
        if min_dist < best_dist:
            second_best = best_dist
            best_dist = min_dist
            best_match = person
        elif min_dist < second_best:
            second_best = min_dist


    if best_match and best_dist < STRICT_TH and (second_best - best_dist) > 0.10:
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

        with lock:
            latest_frames[cam_name] = cv2.resize(frame, (480, 360))

    cap.release()

def dashboard_loop(camera_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 2
    bg_color = (0, 0, 0)

    while True:
        with lock:
            frames = []
            for _, cam_name in camera_list:
                frame = latest_frames.get(cam_name, np.zeros((360, 480, 3), dtype=np.uint8))
                
                cv2.rectangle(frame, (0, 0), (200, 30), bg_color, -1)
                
                cv2.putText(frame, cam_name, (10, 22), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                frames.append(frame)

        row_size = int(np.ceil(np.sqrt(len(frames))))
        rows = []
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
        # (0, "webcam"),
        
    ]

    threads = []
    for path, cam_name in camera_list:
        t = threading.Thread(target=process_camera, args=(path, cam_name))
        t.start()
        threads.append(t)

    dashboard_loop(camera_list)

    for t in threads:
        t.join()