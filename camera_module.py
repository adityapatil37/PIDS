"""
camera_module.py  —  High-Accuracy Edition
-------------------------------------------
Key improvements over previous version:

  MATCHING
  --------
  1. Hybrid distance: cosine (70%) + L2 (30%) — handles scale variation better
  2. Tighter thresholds: STRICT 0.35→0.30, LOOSE 0.55→0.48
  3. Match against mean of top-3 nearest embeddings, not just single minimum
  4. Stricter ratio test (0.75→0.70) — best match must be more distinct

  EMBEDDING QUALITY
  -----------------
  5. Ensemble feature extraction: 3 overlapping crops averaged per frame
  6. Outlier rejection in gallery: discard embeddings that drift too far
  7. Weighted EMA gallery: stable representative embedding, not just latest
  8. Skip tiny boxes (<20x40px) — embeddings from tiny crops are unreliable

  IDENTITY STABILITY
  ------------------
  9. Voting system: require N=4 consecutive strong matches before committing
 10. Identity lock: once conf ≥ 0.75, requires much stronger evidence to override
 11. Re-assignment requires voted_conf > current_conf + 0.20 gap
 12. Clear voter state when a track disappears (prevents ghost assignments)

  TRACK LIFECYCLE
  ---------------
 13. Stale track_info purged every 5s (TTL=10s) — kills ghost identities
 14. Gallery entries removed when track expires — no embedding bleed-over
 15. Reconnect loop re-applies camera resolution settings

  ALERTS
  ------
 16. Unknown grace period increased 5s→8s (fewer false alarms on slow recognition)
 17. Alert cooldown increased 10s→15s per camera
"""

import os
import cv2
import time
import torch
import threading
import numpy as np
import datetime
import re
from collections import deque
from scipy.spatial.distance import cosine
from pymongo import MongoClient

from torchreid.utils import FeatureExtractor

# --- Custom Modules ---
from global_tracker import global_tracker
from core.detectors.robust_yolo import RobustDetector
from core.trackers.robust_tracker import RobustTracker
from core.intelligence.quality_filter import QualityFilter
from core.intelligence.behavior_engine import BehaviorEngine
from core.forensics.audit_ledger import AuditLedger

# ── Configuration ─────────────────────────────────────────────────────────────
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAM_RESOLUTION_W = 640
CAM_RESOLUTION_H = 480
PROCESS_EVERY_N_FRAMES = 1

# ── Matching thresholds ───────────────────────────────────────────────────────
STRICT_TH    = 0.30    # was 0.35 — must be very close to confirm
LOOSE_TH     = 0.48    # was 0.55 — tighter uncertain band
RATIO_MARGIN = 0.70    # was 0.75 — nearest/second-nearest must be more distinct

# ── Identity assignment policy ────────────────────────────────────────────────
REID_INTERVAL         = 2     # run ReID every N frames (more frequent)
MIN_CONF_ASSIGN       = 0.60  # was 0.55 — higher bar to assign a name
CONF_MARGIN           = 0.20  # was 0.18 — bigger gap required to re-assign
CONFIRM_VOTES_NEEDED  = 4     # consecutive strong matches before locking name
REVERT_VOTES_NEEDED   = 6     # non-matching frames before clearing candidate
IDENTITY_LOCK_CONF    = 0.75  # once conf reaches this, lock identity
LOCK_OVERRIDE_MARGIN  = 0.30  # locked identity needs this much better score to override

# ── Track lifecycle ───────────────────────────────────────────────────────────
TRACK_MAX_AGE  = 60    # DeepSort max_age (frames before track dies)
TRACK_N_INIT   = 3     # frames before track is confirmed
TRACK_INFO_TTL = 10.0  # seconds — purge track_info entries older than this

# ── Gallery ───────────────────────────────────────────────────────────────────
GALLERY_EMA_ALPHA      = 0.80
GALLERY_MAX_FRAMES     = 20
GALLERY_OUTLIER_THRESH = 0.55  # discard if embedding too far from EMA

# ── Alert / logging ───────────────────────────────────────────────────────────
UNKNOWN_ALERT_GRACE = 8.0   # seconds before unknown triggers alert (was 5)
ALERT_COOLDOWN      = 15.0  # min seconds between alerts per camera (was 10)
LOG_INTERVAL        = 5

# ── Geometry / zones ─────────────────────────────────────────────────────────
TILE_W, TILE_H   = 640, 480
UNKNOWN_SAVE_DIR = "unknown_crops"
ZONE_CONFIG      = {"RedZone": [100, 100, 300, 300]}

os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)
print(f"[CameraModule] Using device: {DEVICE}")

# ── Database ──────────────────────────────────────────────────────────────────
_mongo_client = MongoClient("mongodb://localhost:27017/")
_db           = _mongo_client["person_reid"]
people_col    = _db["people"]
logs_col      = _db["logs"]
history_col   = _db["track_history"]
access_col    = _db["access_control"]
alerts_col    = _db["alerts"]

# ── Module singletons ─────────────────────────────────────────────────────────
audit_ledger = AuditLedger("secure_audit.jsonl")
detector     = RobustDetector(device=DEVICE)
extractor    = FeatureExtractor(
    model_name=REID_MODEL_NAME,
    model_path=REID_MODEL_PATH,
    device=DEVICE,
)

# ── Shared state (written by camera threads, read by Flask) ───────────────────
latest_frames:   dict = {}
frame_meta:      dict = {}
frames_lock            = threading.Lock()
last_alert_time: dict = {}
last_log_times:  dict = {}


# =============================================================================
# Feature Utilities
# =============================================================================

def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def hybrid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Weighted combination of cosine and normalised L2 distance.
    More robust than cosine alone — handles lighting / scale variation.
    """
    cos_d = float(cosine(a, b))
    l2_d  = float(np.linalg.norm(a - b)) / 2.0   # normalised to ~[0,1]
    return 0.7 * cos_d + 0.3 * l2_d


def extract_feature_from_crop(frame: np.ndarray, box,
                               margin: float = 0.08) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 40:          # skip tiny detections
        return None
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
    x2 = min(w-1, x2 + pad_x); y2 = min(h-1, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat_t = extractor([crop_rgb])
    return l2norm(feat_t[0].cpu().numpy().flatten())


def extract_ensemble_feature(frame: np.ndarray, box) -> np.ndarray | None:
    """
    Average features from 3 overlapping vertical sub-crops.
    Significantly more stable than a single crop extraction.
    Falls back to single crop for small boxes.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h_box = y2 - y1
    if h_box < 80:
        return extract_feature_from_crop(frame, box)

    third = h_box // 3
    sub_boxes = [
        (x1, y1,          x2, y1 + 2 * third),  # top 2/3
        (x1, y1 + third,  x2, y2 - third),       # middle strip
        (x1, y1 + third,  x2, y2),               # bottom 2/3
    ]
    feats = []
    for sb in sub_boxes:
        f = extract_feature_from_crop(frame, sb, margin=0.05)
        if f is not None:
            feats.append(f)
    if not feats:
        return None
    return l2norm(np.mean(feats, axis=0))


# =============================================================================
# SmartGallery — replaces GalleryManager
# =============================================================================

class SmartGallery:
    """
    Per-track embedding store with outlier rejection and EMA smoothing.

    - Outlier rejection: discard embeddings that are too far from current EMA
    - Weighted EMA: new embeddings are blended with a weighted moving average
    - get_embedding() returns the EMA (most stable representative)
    """

    def __init__(self,
                 max_frames:     int   = GALLERY_MAX_FRAMES,
                 ema_alpha:      float = GALLERY_EMA_ALPHA,
                 outlier_thresh: float = GALLERY_OUTLIER_THRESH):
        self._galleries: dict[int, deque]      = {}
        self._ema:       dict[int, np.ndarray] = {}
        self.max_frames     = max_frames
        self.ema_alpha      = ema_alpha
        self.outlier_thresh = outlier_thresh

    def update(self, tid: int, feat: np.ndarray) -> bool:
        """Returns True if embedding was accepted, False if rejected as outlier."""
        feat = l2norm(feat)
        if tid not in self._ema:
            self._galleries[tid] = deque(maxlen=self.max_frames)
            self._ema[tid]       = feat.copy()
            self._galleries[tid].append(feat)
            return True

        dist = hybrid_distance(feat, self._ema[tid])
        if dist > self.outlier_thresh:
            return False   # outlier — discard silently

        self._ema[tid] = l2norm(
            self.ema_alpha * self._ema[tid] + (1 - self.ema_alpha) * feat
        )
        self._galleries[tid].append(feat)
        return True

    def get_embedding(self, tid: int) -> np.ndarray | None:
        return self._ema.get(tid)

    def has(self, tid: int) -> bool:
        return tid in self._ema

    def remove(self, tid: int):
        self._galleries.pop(tid, None)
        self._ema.pop(tid, None)


# =============================================================================
# IdentityVoter — requires N consecutive strong matches before committing
# =============================================================================

class IdentityVoter:
    """
    Accumulate evidence across frames before committing to an identity.
    Prevents single-frame mismatches from changing a confirmed identity.
    """

    def __init__(self,
                 confirm_votes: int = CONFIRM_VOTES_NEEDED,
                 revert_votes:  int = REVERT_VOTES_NEEDED):
        self.confirm_votes = confirm_votes
        self.revert_votes  = revert_votes
        self._state: dict[int, dict] = {}

    def vote(self, tid: int, name, role, conf: float, strong: bool):
        """
        Submit one frame's match result.
        Returns (name, role, conf, is_confirmed) once threshold is reached,
        otherwise (None, None, 0.0, False).
        """
        s = self._state.setdefault(tid, {
            "candidate":    None,
            "role":         None,
            "count":        0,
            "conf":         0.0,
            "revert_count": 0,
        })

        if name is None or not strong:
            s["revert_count"] += 1
            s["count"] = max(0, s["count"] - 1)
            if s["revert_count"] >= self.revert_votes:
                s.update({"candidate": None, "role": None,
                           "count": 0, "conf": 0.0, "revert_count": 0})
            return None, None, 0.0, False

        # Strong match received
        s["revert_count"] = 0
        if s["candidate"] != name:
            s.update({"candidate": name, "role": role, "count": 1, "conf": conf})
        else:
            s["count"] += 1
            s["conf"]   = max(s["conf"], conf)

        if s["count"] >= self.confirm_votes:
            return s["candidate"], s["role"], s["conf"], True

        return None, None, 0.0, False

    def reset(self, tid: int):
        self._state.pop(tid, None)


# =============================================================================
# Database helpers
# =============================================================================

def load_known_people() -> list:
    people = []
    for doc in people_col.find():
        feats = []
        for f in doc.get("features", []):
            arr = np.array(f, dtype=np.float32).flatten()
            if arr.size > 0:
                feats.append(l2norm(arr))
        if feats:
            people.append({
                "name":     doc["name"],
                "role":     doc.get("role", "Unknown"),
                "features": feats,
            })
    print(f"[DB] Loaded {len(people)} known people")
    return people


def match_person(feat: np.ndarray, known_people: list):
    """
    Hybrid-distance match with ratio test.
    Uses mean of top-3 distances per person (more stable than single min).
    Returns (name, role, dist, is_strong_match).
    """
    if feat is None or not known_people:
        return None, None, None, False

    f = l2norm(feat)
    best_person, best_dist, second_dist = None, 1.0, 1.0

    for person in known_people:
        dists = [hybrid_distance(f, ex) for ex in person["features"]]
        if not dists:
            continue
        dists_sorted = sorted(dists)
        # Average top-3 closest matches (guards against one lucky outlier)
        min_d = float(np.mean(dists_sorted[:min(3, len(dists_sorted))]))

        if min_d < best_dist:
            second_dist, best_dist, best_person = best_dist, min_d, person
        elif min_d < second_dist:
            second_dist = min_d

    if best_person is None:
        return None, None, None, False

    ratio    = best_dist / (second_dist + 1e-12)
    ratio_ok = ratio < RATIO_MARGIN
    strong   = (best_dist < STRICT_TH) and ratio_ok
    weak     = (best_dist < LOOSE_TH)  and ratio_ok

    if strong:
        return best_person["name"], best_person["role"], best_dist, True
    if weak:
        return best_person["name"] + " (?)", best_person["role"], best_dist, False
    return None, None, None, False


def dist_to_conf(dist) -> float:
    if dist is None:      return 0.0
    if dist <= STRICT_TH: return 1.0
    if dist >= LOOSE_TH:  return 0.0
    return float((LOOSE_TH - dist) / (LOOSE_TH - STRICT_TH))


# =============================================================================
# Alerting / Logging
# =============================================================================

def _play_sound():
    try:
        import subprocess
        for _ in range(5):
            subprocess.run(["beep"], check=False)
            time.sleep(0.1)
    except Exception:
        pass


def trigger_alert(frame: np.ndarray, cam_name: str, tid: int):
    ts_dt = datetime.datetime.now()
    ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")

    snap_dir = os.path.join("static", "alert_snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    snap_name = f"{cam_name}_{tid}_{int(time.time())}.jpg"
    snap_path = os.path.join(snap_dir, snap_name)

    cv2.imwrite(snap_path, frame)

    print(f"🚨 [ALERT] UNKNOWN on {cam_name} (track {tid}) at {ts_str}")

    # Persist to MongoDB so frontend /alerts and /camera/recent_alerts can show it
    alerts_col.insert_one({
        "timestamp": ts_dt,
        "person_name": "UNKNOWN",
        "camera_name": cam_name,
        "status": "Unknown Person Detected",
        "track_id": tid,
        "thumbnail": snap_path,
    })

    audit_ledger.log(
        "ALERT_UNKNOWN",
        {
            "camera": cam_name,
            "track_id": tid,
            "snapshot": snap_path
        }
    )

    # Live sidebar / camera-status support
    with frames_lock:
        meta = frame_meta.setdefault(cam_name, {})
        meta.setdefault("alerts", []).append({
            "type": "UNKNOWN",
            "track_id": tid,
            "time": ts_str,
            "thumbnail": snap_path,
        })

        # keep only recent alerts in memory
        meta["alerts"] = meta["alerts"][-10:]

    threading.Thread(target=_play_sound, daemon=True).start()

def trigger_zone_alert(name: str, cam_name: str, frame: np.ndarray, bbox):
    ts = datetime.datetime.now()
    snap_dir = os.path.join("static", "alert_snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]

    thumb_name = f"{cam_name}_{name}_{int(time.time())}.jpg"
    thumb_path = os.path.join("alert_snapshots", thumb_name)
    cv2.imwrite(thumb_path, crop)

    alerts_col.insert_one({
        "timestamp": ts,
        "person_name": name,
        "camera_name": cam_name,
        "status": "Unauthorized Zone Entry",
        "thumbnail": thumb_path,
    })

    audit_ledger.log(
        "ALERT_ZONE",
        {
            "person": name,
            "camera": cam_name,
            "thumbnail": thumb_path
        }
    )

    print(f"🚫 [ZONE ALERT] {name} entered restricted camera: {cam_name} at {ts}")
    threading.Thread(target=_play_sound, daemon=True).start()


def log_person_event(name: str, cam_name: str, tid: int,
                     frame: np.ndarray, bbox):
    now = time.time()
    key = (cam_name, tid, name)
    if key in last_log_times and now - last_log_times[key] < LOG_INTERVAL:
        return
    last_log_times[key] = now
    ts = datetime.datetime.now()
    x, y, w, h = bbox
    x1, y1 = max(0, x),                max(0, y)
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    crop = frame[y1:y2, x1:x2]
    thumb_dir  = os.path.join("static", "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_name = f"{cam_name}_{tid}_{int(time.time())}.jpg"
    thumb_path = os.path.join(thumb_dir, thumb_name)
    cv2.imwrite(thumb_path, crop)
    web_path = f"/static/thumbnails/{thumb_name}"
    history_col.insert_one({
        "timestamp": ts, "person_name": name,
        "camera_name": cam_name, "track_id": tid, "thumbnail": web_path,
    })
    audit_ledger.log("PERSON_DETECTED",
                     {"person": name, "camera": cam_name, "track_id": tid})
    print(f"📋 [DB] Logged {name} on {cam_name}, track {tid} at {ts}")


def check_access_permission(person_name: str, cam_name: str,
                             frame: np.ndarray, bbox):
    if (not person_name or person_name == "UNKNOWN"
            or re.search(r"\s*\(\?\)$", person_name.strip())):
        return
    doc     = access_col.find_one({"camera_name": cam_name})
    allowed = doc.get("allowed_people", ["*"]) if doc else ["*"]
    if allowed != ["*"] and person_name not in allowed:
        trigger_zone_alert(person_name, cam_name, frame, bbox)


# =============================================================================
# Per-camera processing thread
# =============================================================================

def process_camera(source, cam_name: str, known_reload_interval: int = 30):
    global last_alert_time

    print(f"[{cam_name}] starting — source={source!r}")
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_RESOLUTION_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION_H)

    tracker        = RobustTracker(max_age=TRACK_MAX_AGE, n_init=TRACK_N_INIT)
    quality_filter = QualityFilter()
    gallery        = SmartGallery()
    voter          = IdentityVoter()
    behavior       = BehaviorEngine(zone_config=ZONE_CONFIG)

    track_info: dict[int, dict] = {}
    known_people    = load_known_people()
    last_known_load = time.time()
    last_purge_time = time.time()
    frame_idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] stream lost — retrying…")
            with frames_lock:
                latest_frames[cam_name] = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(source)
            if isinstance(source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_RESOLUTION_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION_H)
            continue

        frame_idx += 1

        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            with frames_lock:
                latest_frames[cam_name] = cv2.resize(frame, (TILE_W, TILE_H))
            time.sleep(0.01)
            continue

        # ── Reload known people ───────────────────────────────────────────
        if time.time() - last_known_load > known_reload_interval:
            known_people    = load_known_people()
            last_known_load = time.time()

        # ── Purge stale track_info (prevents ghost identity bleed) ────────
        if time.time() - last_purge_time > 5.0:
            now_t = time.time()
            stale = [
                tid for tid, info in track_info.items()
                if now_t - info.get("last_seen", 0) > TRACK_INFO_TTL
            ]
            for tid in stale:
                track_info.pop(tid, None)
                gallery.remove(tid)
                voter.reset(tid)
            if stale:
                print(f"[{cam_name}] Purged {len(stale)} stale tracks: {stale}")
            last_purge_time = time.time()

        # ── 1. Detect (full resolution for accuracy) ──────────────────────
        detections = detector.detect(frame, img_size=640)

        # ── 2. Track ──────────────────────────────────────────────────────
        tracks = tracker.update(detections, frame=frame)

        # ── 3. Behavior ───────────────────────────────────────────────────
        events = behavior.update(tracks)
        for event in events:
            if event["type"] == "ZONE_CHANGE":
                new_zone = event["to"]
                tid      = event["track_id"]
                if new_zone not in ("General", "None"):
                    info = track_info.get(tid)
                    name = info["name"] if info else "UNKNOWN"
                    print(f"🚫 [ZONE ENTRY] {name} → {new_zone} on {cam_name}")

        # ── 4. Identity ───────────────────────────────────────────────────
        people_in_frame = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb            = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            tid             = track.track_id

            # Initialise track info
            if tid not in track_info:
                track_info[tid] = {
                    "name":            "UNKNOWN",
                    "role":            "",
                    "conf":            0.0,
                    "locked":          False,
                    "last_reid_frame": -999,
                    "last_seen":       time.time(),
                    "first_seen":      time.time(),
                }
            info = track_info[tid]
            info["last_seen"] = time.time()

            # ── Feature extraction ────────────────────────────────────────
            should_run_reid = (frame_idx - info["last_reid_frame"]) >= REID_INTERVAL
            if should_run_reid:
                is_good, _ = quality_filter.check(
                    frame, (x1, y1, x2 - x1, y2 - y1)
                )
                if is_good:
                    feat_raw = extract_ensemble_feature(frame, (x1, y1, x2, y2))
                    if feat_raw is not None:
                        accepted = gallery.update(tid, feat_raw)
                        if accepted:
                            info["last_reid_frame"] = frame_idx

            feat = gallery.get_embedding(tid)

            # ── Matching & identity update ────────────────────────────────
            if feat is not None:
                global_id = global_tracker.update_track(
                    cam_name, tid, feat, (x1, y1, x2, y2)
                )
                info["global_id"] = global_id

                name, role, dist, strong = match_person(feat, known_people)
                new_conf = dist_to_conf(dist)

                if info["locked"]:
                    # ── Locked: only override with significantly better match ──
                    if name == info["name"] and strong:
                        # Reinforce (soft EMA update)
                        info["conf"] = min(1.0, info["conf"] * 0.97 + new_conf * 0.03)
                    elif strong and new_conf > info["conf"] + LOCK_OVERRIDE_MARGIN:
                        print(f"[{cam_name}] 🔓 Override TID={tid}: "
                              f"{info['name']} → {name} "
                              f"(Δ={new_conf - info['conf']:.2f})")
                        info.update({"name": name, "role": role,
                                     "conf": new_conf, "locked": False})
                        voter.reset(tid)
                else:
                    # ── Voting phase ──────────────────────────────────────
                    voted_name, voted_role, voted_conf, confirmed = voter.vote(
                        tid, name, role, new_conf, strong
                    )

                    if confirmed:
                        if info["name"] == "UNKNOWN":
                            info.update({"name": voted_name, "role": voted_role,
                                         "conf": voted_conf})
                            print(f"[{cam_name}] ✅ Confirmed TID={tid}: "
                                  f"{voted_name} ({voted_conf:.0%})")
                        elif voted_name != info["name"]:
                            if voted_conf > info["conf"] + CONF_MARGIN:
                                print(f"[{cam_name}] 🔄 Re-assigned TID={tid}: "
                                      f"{info['name']} → {voted_name}")
                                info.update({"name": voted_name, "role": voted_role,
                                             "conf": voted_conf})
                        else:
                            info["conf"] = max(info["conf"], voted_conf)

                        # Lock if high confidence
                        if info["conf"] >= IDENTITY_LOCK_CONF and not info["locked"]:
                            info["locked"] = True
                            print(f"[{cam_name}] 🔒 Locked TID={tid}: "
                                  f"{info['name']} ({info['conf']:.0%})")

            # ── Unknown alert ─────────────────────────────────────────────
            if info["name"] == "UNKNOWN":
                elapsed     = time.time() - info.get("first_seen", time.time())
                cooldown_ok = (time.time() - last_alert_time.get(cam_name, 0)
                               > ALERT_COOLDOWN)
                if elapsed >= UNKNOWN_ALERT_GRACE and cooldown_ok:
                    trigger_alert(frame, cam_name, tid)
                    last_alert_time[cam_name] = time.time()

            # ── Draw ──────────────────────────────────────────────────────
            display_name = info["name"]
            is_unknown   = display_name == "UNKNOWN"
            is_uncertain = " (?)" in display_name

            if is_unknown:
                color = (0, 0, 255)
            elif is_uncertain:
                color = (0, 165, 255)
            else:
                color = (0, 200, 0)

            thickness = 3 if info.get("locked") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            conf_pct = int(info["conf"] * 100)
            label    = (display_name if is_unknown
                        else f"{display_name} {conf_pct}%")

            # Filled label background for readability
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            lx, ly = x1, max(lh + 6, y1 - 2)
            cv2.rectangle(frame,
                          (lx, ly - lh - 4), (lx + lw + 6, ly + 2),
                          color, -1)
            cv2.putText(frame, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Log and access check
            if not is_unknown and not is_uncertain:
                bbox_wh = (x1, y1, x2 - x1, y2 - y1)
                log_person_event(display_name, cam_name, tid, frame, bbox_wh)
                check_access_permission(display_name, cam_name, frame, bbox_wh)

            people_in_frame.append({
                "track_id": tid,
                "name":     display_name,
                "role":     info.get("role", ""),
                "conf":     round(info.get("conf", 0.0), 2),
                "locked":   info.get("locked", False),
            })

        # ── Write shared state ─────────────────────────────────────────────
        resized = cv2.resize(frame, (TILE_W, TILE_H))
        with frames_lock:
            latest_frames[cam_name] = resized
            frame_meta[cam_name]    = {
                "people":     people_in_frame,
                "count":      len(people_in_frame),
                "updated_at": time.time(),
            }

    cap.release()
    print(f"[{cam_name}] thread exiting")


# =============================================================================
# MJPEG generator (used by Flask /camera/stream/<cam_name>)
# =============================================================================

def gen_frames(cam_name: str):
    while True:
        with frames_lock:
            frame = latest_frames.get(cam_name)

        if frame is None:
            blank = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            cv2.putText(blank, "Connecting...", (20, TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
            frame = blank

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )
        time.sleep(1 / 25)


# =============================================================================
# Launcher
# =============================================================================

def start_cameras(camera_list: list) -> list:
    threads = []
    for src, cam in camera_list:
        t = threading.Thread(target=process_camera, args=(src, cam), daemon=True)
        t.start()
        threads.append(t)
        print(f"[Launcher] Started thread for {cam!r}")
    return threads