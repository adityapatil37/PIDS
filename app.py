import cv2
import threading
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from collections import deque
import os
import time

YOLO_WEIGHTS = "yolov12n.pt"   # or "yolov11n.pt"
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STRICT_TH = 0.35
CROSSCAM_TH = 0.65     # looser for cross-camera
LOCAL_EMB_HISTORY = 10 # smoother local embeddings
GLOBAL_EMA_ALPHA = 0.9 # stronger memory
GLOBAL_EXPIRE_SEC = 120

print("Device:", DEVICE)
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to(DEVICE)

extractor = FeatureExtractor(
    model_name=REID_MODEL_NAME,
    model_path=REID_MODEL_PATH,
    device=DEVICE
)

global_gallery = {}   # gid -> {"feat": np.array, "last_seen": timestamp, "cam": str}
next_global_id = 0
global_lock = threading.Lock()

local_emb_hist = {}
local_emb_lock = threading.Lock()

def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

def safe_crop(frame, bbox, pad=5):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def batch_extract_reid_features(frame, boxes):
    crops, idxs = [], []
    for i, box in enumerate(boxes):
        c = safe_crop(frame, box)
        if c is not None and c.size > 0:
            crops.append(c)
            idxs.append(i)
        else:
            crops.append(None)

    if not idxs:
        return [None] * len(boxes)

    rgb_crops = [cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB) for i in idxs]
    with torch.no_grad():
        feats_t = extractor(rgb_crops)
    feats = feats_t.cpu().numpy()

    out = [None] * len(boxes)
    for j, i_orig in enumerate(idxs):
        out[i_orig] = l2_normalize(feats[j].flatten())
    return out

def cleanup_global_gallery(timeout=GLOBAL_EXPIRE_SEC):
    now = time.time()
    with global_lock:
        expired = [gid for gid, v in global_gallery.items() if now - v["last_seen"] > timeout]
        for gid in expired:
            del global_gallery[gid]

def match_global_id(avg_feature, cam_name, current_gids_in_frame,
                    strict_th=STRICT_TH, crosscam_th=CROSSCAM_TH):
    global next_global_id
    now = time.time()
    avg_feature = l2_normalize(avg_feature)

    with global_lock:
        best_gid, best_dist, best_cam = None, 1.0, None
        for gid, entry in global_gallery.items():
            try:
                d = float(cosine(avg_feature, entry["feat"]))
            except Exception:
                continue
            if d < best_dist:
                best_dist, best_gid, best_cam = d, gid, entry["cam"]

        if best_gid is not None:
            if best_cam == cam_name:
                if best_gid in current_gids_in_frame:
                    best_gid = None
                elif best_dist < strict_th:
                    global_gallery[best_gid]["feat"] = l2_normalize(
                        GLOBAL_EMA_ALPHA * global_gallery[best_gid]["feat"] +
                        (1.0 - GLOBAL_EMA_ALPHA) * avg_feature
                    )
                    global_gallery[best_gid]["last_seen"] = now
                    return best_gid
            else:
                if best_dist < crosscam_th:
                    global_gallery[best_gid]["feat"] = l2_normalize(
                        GLOBAL_EMA_ALPHA * global_gallery[best_gid]["feat"] +
                        (1.0 - GLOBAL_EMA_ALPHA) * avg_feature
                    )
                    global_gallery[best_gid]["last_seen"] = now
                    global_gallery[best_gid]["cam"] = cam_name
                    return best_gid

        gid = next_global_id
        global_gallery[gid] = {
            "feat": avg_feature.copy(),
            "last_seen": now,
            "cam": cam_name
        }
        next_global_id += 1
        return gid

def process_camera(cam_url, cam_name):
    print(f"[{cam_name}] starting, source={cam_url}")
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)
    cap = cv2.VideoCapture(cam_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] stream ended")
            break

        results = yolo_model(frame, verbose=False)

        detections, boxes_for_reid, local_track_ids = [], [], []
        for r in results[0].boxes:
            cls_id, conf = int(r.cls[0]), float(r.conf[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            local_id = track.track_id
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
            boxes_for_reid.append((x1, y1, x2, y2))
            local_track_ids.append(local_id)

        features = batch_extract_reid_features(frame, boxes_for_reid) if boxes_for_reid else []
        current_gids_in_frame = set()

        for local_id, box, feat in zip(local_track_ids, boxes_for_reid, features):
            if feat is None:
                continue
            key = (cam_name, local_id)
            with local_emb_lock:
                if key not in local_emb_hist:
                    local_emb_hist[key] = deque(maxlen=LOCAL_EMB_HISTORY)
                local_emb_hist[key].append(feat.copy())
                avg_feat = np.mean(np.vstack(local_emb_hist[key]), axis=0)

            gid = match_global_id(avg_feat, cam_name, current_gids_in_frame)
            current_gids_in_frame.add(gid)

            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"GID {gid}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(cam_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow(cam_name)
    print(f"[{cam_name}] exiting")

if __name__ == "__main__":
    camera_streams = [
        "vid3.mp4",
        "vid4.mp4"
    ]

    threads = []
    for i, cam_url in enumerate(camera_streams):
        t = threading.Thread(target=process_camera, args=(cam_url, f"Camera_{i}"), daemon=True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
