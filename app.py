import cv2
import queue
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import torch
from scipy.spatial.distance import cosine
import os
import time
import threading

# Fix for OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv12 model
det_model = YOLO("yolov12n.pt")
det_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load ReID FeatureExtractor (light model for speed)
extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x0_25_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Global track manager: {track_id: (camera_id, last_frame_time, reid_feature)}
global_tracks = {}
global_id_counter = 0
lock = threading.Lock()

# Tracking parameters
params = {
    "conf": 0.25,
    "iou": 0.45,
    "classes": 0,  # Persons only
    "persist": True,
    "tracker": "botsort.yaml",  # Ensure with_reid: True
    "imgsz": 640  # Reduced for speed
}

def extract_reid_features(frame, bboxes):
    """Batch extract ReID features for multiple bounding boxes."""
    crops = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in bboxes]
    crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops if c.size > 0]
    if not crops:
        return [None] * len(bboxes)
    features = extractor(crops).cpu().numpy()
    return [f.flatten() if f is not None else None for f in features]

def match_global(feature, timestamp, threshold=0.5):
    """Match to global tracks."""
    global global_id_counter
    with lock:
        for gid, (cid, last_t, g_feature) in list(global_tracks.items()):
            if timestamp - last_t > 30:
                del global_tracks[gid]
                continue
            if cosine(feature, g_feature) < threshold:
                return gid
        global_id_counter += 1
        return global_id_counter

# Initialize video sources
sources = ["vid2.mp4", "vid1.mp4"]
caps = []
widths, heights, fps_list = [], [], []
for source in sources:
    if not os.path.exists(source):
        print(f"Error: Video file {source} not found")
        exit(1)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Video file {source} cannot be opened")
        exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    if width == 0 or height == 0:
        print(f"Error: Invalid dimensions for {source}")
        cap.release()
        exit(1)
    caps.append(cap)
    widths.append(width)
    heights.append(height)
    fps_list.append(fps)
    print(f"Initialized {source}: {width}x{height}, {fps} FPS")

# Set up output video (side-by-side)
max_height = max(heights)
total_width = sum(int(w * max_height / h) for w, h in zip(widths, heights))
out_fps = min(fps_list)  # Use lowest FPS for consistency
out = cv2.VideoWriter(
    "output_combined.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    out_fps,
    (total_width, max_height)
)

# Process frames consecutively
local_tracks = [{} for _ in sources]  # Per-camera tracks
frame_nums = [0] * len(sources)
start_time = time.time()
all_done = False

while not all_done:
    all_done = True
    combined_frame = None
    for cam_id, (cap, source, height, width) in enumerate(zip(caps, sources, heights, widths), 1):
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_id} ({source}): End of video")
            continue
        all_done = False
        frame_nums[cam_id - 1] += 1

        # Skip every other frame for speed (adjust as needed)
        if frame_nums[cam_id - 1] % 2 == 0:
            continue

        try:
            results = det_model.track(source=frame, **params)
        except Exception as e:
            print(f"Camera {cam_id} ({source}): Tracking error - {e}")
            continue

        bboxes = [track.xyxy[0] for track in results[0].boxes if track.id is not None]
        features = extract_reid_features(frame, bboxes) if bboxes else []
        for track, feature in zip(results[0].boxes, features):
            if track.id is None or feature is None:
                continue
            local_id = int(track.id)
            bbox = track.xyxy[0]
            if local_id not in local_tracks[cam_id - 1]:
                global_id = match_global(feature, frame_nums[cam_id - 1] / fps_list[cam_id - 1])
                local_tracks[cam_id - 1][local_id] = (global_id, feature)
            else:
                global_id, _ = local_tracks[cam_id - 1][local_id]
                with lock:
                    if global_id in global_tracks:
                        old_feature = global_tracks[global_id][2]
                        new_feature = (old_feature + feature) / 2
                        global_tracks[global_id] = (cam_id, frame_nums[cam_id - 1] / fps_list[cam_id - 1], new_feature)

            cv2.putText(
                frame,
                f"GID: {global_id}",
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        annotated_frame = results[0].plot()
        scale = max_height / height
        resized_frame = cv2.resize(annotated_frame, (int(width * scale), max_height))
        cv2.putText(resized_frame, f"Cam {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if combined_frame is None:
            combined_frame = resized_frame
        else:
            combined_frame = np.hstack((combined_frame, resized_frame))

        print(f"Camera {cam_id} ({source}): Processed frame {frame_nums[cam_id - 1]}, CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    if combined_frame is not None:
        out.write(combined_frame)
        cv2.imshow("Multi-Cam Tracking", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup
for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
print(f"Multi-camera tracking complete. Output saved to output_combined.mp4")
print(f"Total FPS: {sum(frame_nums) / elapsed:.2f}")