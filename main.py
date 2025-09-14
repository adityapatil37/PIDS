import cv2
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv12 model (replace with your model variant, e.g., yolov12m.pt for medium)
model = YOLO("yolov12m.pt")

# Input video source (replace with your video file path or 0 for webcam)
# source = "test.mp4"  # Or 0 for real-time webcam
source = 0

# Output video settings
cap = cv2.VideoCapture(source)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Tracking parameters
# conf: Minimum confidence for detections (lower for better recall in occlusions)
# iou: IoU threshold for non-max suppression
# classes: 0 for persons only (COCO class ID)
# persist: Enable persistent tracking across frames
# tracker: Use ByteTrack configuration
params = {
    "conf": 0.1,      # Adjust based on your needs (e.g., 0.1 for more detections)
    "iou": 0.45,
    "classes": 0,      # Track only persons
    "persist": True,
    "tracker": "bytetrack.yaml"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run tracking
    results = model.track(source=frame, **params)
    
    # Visualize: Draw bounding boxes, IDs, and confidence
    annotated_frame = results[0].plot()
    
    # Write to output video
    out.write(annotated_frame)
    
    # Optional: Display in real-time (comment out if not needed)
    cv2.imshow("Person Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Tracking complete. Output saved to 'output_tracked.mp4'")