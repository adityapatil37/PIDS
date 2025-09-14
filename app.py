import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import torch
from scipy.spatial.distance import cosine
import os

# Fix for OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv12 model
det_model = YOLO("yolov12n.pt")

# Load ReID FeatureExtractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Global track manager: {track_id: (camera_id, last_frame, reid_feature)}
global_tracks = {}
global_id_counter = 0
lock = threading.Lock()

# Tracking parameters (use botsort for ReID)
params = {
    "conf": 0.25,
    "iou": 0.45,
    "classes": 0,  # Persons only
    "persist": True,
    "tracker": "botsort.yaml"  # Ensure botsort.yaml has with_reid: True
}

def extract_reid_feature(frame, bbox):
    """Crop and extract ReID feature from bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    feature = extractor(crop).cpu().numpy().flatten()
    return feature

def match_global(feature, timestamp, threshold=0.5):
    """Match to existing global tracks based on ReID similarity and time."""
    global global_id_counter
    with lock:
        for gid, (cid, last_t, g_feature) in list(global_tracks.items()):
            if timestamp - last_t > 30:  # Prune old tracks (30s)
                del global_tracks[gid]
                continue
            if cosine(feature, g_feature) < threshold:
                return gid
        global_id_counter += 1
        return global_id_counter

def process_camera(source, cam_id, output_queue):
    """Process a single camera feed."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source} for camera {cam_id}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback to 30 if FPS is invalid
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        print(f"Error: Invalid dimensions for camera {cam_id}")
        cap.release()
        return

    out = cv2.VideoWriter(
        f"output_cam{cam_id}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    print(f"Camera {cam_id} initialized: {source}, {width}x{height}, {fps} FPS")

    local_tracks = {}  # {local_id: (global_id, last_feature)}
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_id}: End of video or error reading frame")
            break
        frame_num += 1

        # Run detection and tracking
        try:
            results = det_model.track(source=frame, **params)
        except Exception as e:
            print(f"Camera {cam_id}: Tracking error - {e}")
            break

        # Process tracks and assign global IDs
        for track in results[0].boxes:
            if track.id is None:
                continue
            local_id = int(track.id)
            bbox = track.xyxy[0]
            feature = extract_reid_feature(frame, bbox)
            if feature is None:
                continue

            if local_id not in local_tracks:
                global_id = match_global(feature, frame_num / fps)
                local_tracks[local_id] = (global_id, feature)
            else:
                global_id, _ = local_tracks[local_id]
                with lock:
                    if global_id in global_tracks:
                        old_feature = global_tracks[global_id][2]
                        new_feature = (old_feature + feature) / 2
                        global_tracks[global_id] = (cam_id, frame_num / fps, new_feature)

            # Draw global ID
            cv2.putText(
                frame,
                f"GID: {global_id}",
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Annotated frame
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        output_queue.put((cam_id, annotated_frame))  # Include cam_id for identification

    cap.release()
    out.release()
    print(f"Camera {cam_id}: Processing complete. Output saved to output_cam{cam_id}.mp4")

# Main: Multiple sources
sources = ["vid2.mp4", "vid1.mp4"]  # Ensure these files exist in the script's directory
threads = []
output_queue = queue.Queue()

# Verify sources exist
for source in sources:
    if not os.path.exists(source):
        print(f"Error: Video file {source} not found")
        exit(1)

# Start threads
for i, source in enumerate(sources):
    t = threading.Thread(target=process_camera, args=(source, i + 1, output_queue))
    t.start()
    threads.append(t)

# Display frames side-by-side
frame_buffers = {}  # {cam_id: latest_frame}
while any(t.is_alive() for t in threads):
    try:
        cam_id, frame = output_queue.get(timeout=1)
        frame_buffers[cam_id] = frame

        # Combine frames side-by-side
        max_height = max((frame.shape[0] for frame in frame_buffers.values()), default=480)
        combined = None
        for cid in sorted(frame_buffers.keys()):
            frame = frame_buffers[cid]
            # Resize to match height
            scale = max_height / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), max_height))
            if combined is None:
                combined = frame
            else:
                combined = np.hstack((combined, frame))

        if combined is not None:
            cv2.imshow("Multi-Cam Tracking", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except queue.Empty:
        pass

# Cleanup
for t in threads:
    t.join()
cv2.destroyAllWindows()

print("Multi-camera tracking complete. Outputs saved as output_cam*.mp4")