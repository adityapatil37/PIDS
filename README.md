# Advanced Multi-Camera Surveillance with Access Control

   

This project is a real-time, multi-camera surveillance system that detects, tracks, and re-identifies people across different video streams. It now includes a **zone-based access control** feature, which triggers alerts if a recognized person enters a restricted area. The system leverages state-of-the-art deep learning models for high accuracy and provides a robust alerting system for both unknown individuals and unauthorized zone entries.

## 🚀 Key Features

  * **Multi-Camera Support**: Processes multiple video streams (from files or live cameras) concurrently using threading.
  * **Person Detection**: Utilizes a **YOLO** model for fast and accurate person detection.
  * **Object Tracking**: Employs **DeepSORT** to assign a stable track ID to each detected person, tracking them as they move within a camera's view.
  * **Person Re-Identification (Re-ID)**: Uses a deep learning Re-ID model (**OSNet**) to extract a unique feature vector (embedding) for each person, enabling recognition across sessions and cameras.
  * **Database Integration**: Connects to a **MongoDB** database to store feature vectors of known individuals and access control rules.
  * **Dual Alert System**:
      * **Unknown Person Alert**: Triggers an audible beep, a desktop pop-up, and saves a snapshot if an **unknown person** remains in view for a configurable duration.
      * **Unauthorized Access Alert**: Triggers a specific alert if a **known person** enters a camera zone where they do not have permission, based on rules in the database.
  * **Historical & Alert Logging**:
      * Logs every confirmed sighting of a *known* person to a `track_history` collection with a cropped thumbnail.
      * Logs all unauthorized zone entries to a dedicated `alerts` collection for auditing.
  * **Centralized Dashboard**: Displays all camera feeds in a single, dynamically sized grid view for easy monitoring.

-----

## 🛠️ How It Works (System Architecture)

The pipeline for each camera stream operates as follows:

1.  **Frame Capture**: The system reads a frame from the video source.
2.  **Detection & Tracking**: The frame is passed to **YOLO** to find people, and the detections are fed into **DeepSORT** to assign and maintain stable track IDs.
3.  **Re-Identification**: For each tracked person, the system periodically extracts a feature vector using the **OSNet (Re-ID) model**. This vector is then compared against the enrolled profiles in the **MongoDB `people` collection**.
4.  **Identification & Logic**:
      * A person is identified with high confidence ("John Doe") if the similarity score passes a strict threshold.
      * A person is identified with low confidence ("John Doe (?)") if the score is weaker. These uncertain matches are ignored for logging and access control checks.
      * If no match is found, the person is labeled "UNKNOWN".
5.  **Access Control & Alerting**:
      * If a person is identified with **high confidence**:
          * The system checks the **`access_control` collection** in MongoDB to see if this person is on the "allowed list" for the current camera.
          * If they are **not allowed**, a zone alert is triggered.
          * If they **are allowed**, their presence is logged to the `track_history` collection.
      * If a track remains "UNKNOWN" for more than `ALERT_UNKNOWN_SECONDS`, a standard unknown person alert is triggered.
6.  **Visualization**: The processed frame, with bounding boxes and identity labels, is sent to the main dashboard for display.

-----

## ⚙️ Tech Stack

  * **AI / ML**: `PyTorch`, `ultralytics` (YOLO), `torchreid` (OSNet), `deep-sort-realtime`
  * **Database**: `MongoDB`, `pymongo`
  * **Core & Utility**: `OpenCV-Python`, `NumPy`, `SciPy`, `tkinter`

-----

## 🏁 Getting Started

### Prerequisites

  * Python 3.8+
  * MongoDB installed and running on `localhost:27017`.
  * An NVIDIA GPU with CUDA is **highly recommended** for real-time performance.

### 1\. Clone the Repository

```bash
git clone https://github.com/adityapatil37/PIDS
cd PIDS
```

### 2\. Create a Virtual Environment & Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate | On macOS/Linux: source venv/bin/activate

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install ultralytics deep-sort-realtime torchreid opencv-python pymongo scipy
```

### 3\. Set Up the MongoDB Database

You need to create the database (`person_reid`) and set up three essential collections.

1.  **`people` Collection**: Stores profiles of known individuals.

      * **Example Document**:
        ```json
        {
          "name": "Jane Doe",
          "role": "Developer",
          "features": [ /* Array of 512-dim feature vectors from enrollment */ ]
        }
        ```

2.  **`access_control` Collection**: Defines which people can access which cameras.

      * **Example Document**:
        ```json
        {
          "camera_name": "Camera_1",
          "allowed_people": ["Jane Doe", "Admin User"]
        }
        ```

    > **Note**: If a camera is not listed in this collection, everyone is implicitly denied access.

3.  **`track_history` & `alerts` Collections**: These will be created and populated automatically by the script as events occur.

### 4\. Configure the Application

Open `app.py` and modify these sections:

  * **Camera Sources**: Update the `camera_list` at the bottom with your video file paths or camera indices.

    ```python
    if __name__ == "__main__":
        camera_list = [
            ("vid3.mp4", "Camera_1"),
            ("vid4.mp4", "Camera_2"),
        ]
    ```

  * **Tuning Parameters**: Adjust thresholds at the top of the file.

    ```python
    STRICT_TH = 0.35            # Lower for more strict matching
    YOLO_CONF_TH = 0.45         # Minimum confidence for a person detection
    ALERT_UNKNOWN_SECONDS = 5   # Time an unknown person must be visible for an alert
    ```

### 5\. Run the System

Execute the script from your terminal:

```bash
python app.py
```

The application will start, connect to the database, process the camera feeds, and open a dashboard window. Press **'q'** with the dashboard window in focus to quit.

-----

## 📁 Project Structure

```
.
├── ids.py                  # Main application script
├── alert_snapshots/        # Directory for alert snapshot images (auto-created)
├── static/
│   └── thumbnails/         # Directory for historical log thumbnails (auto-created)
└── unknown_crops/          # Directory for unknown person thumbnails (auto-created)
```

---
## 📜 License & Acknowledgements
Copyright © 2025 Aditya Patil. All rights reserved.
This project and its contents may not be used, copied, modified, or distributed without explicit permission.