# Multi-Camera Person Re-Identification System with Real-time Alerts

   

This project is a real-time, multi-camera surveillance system that detects, tracks, and re-identifies people across different video streams. It leverages state-of-the-art deep learning models for high accuracy and includes a real-time alerting system for unknown individuals.

## 🚀 Key Features

  * **Multi-Camera Support**: Processes multiple video streams (from files or live cameras) concurrently using threading.
  * **Person Detection**: Utilizes a **YOLO** model for fast and accurate person detection in each frame.
  * **Object Tracking**: Employs **DeepSORT** to assign a stable track ID to each detected person, tracking them as they move within a camera's view.
  * **Person Re-Identification (Re-ID)**: Uses a deep learning Re-ID model (**OSNet**) to extract a unique feature vector (embedding) for each person. This allows the system to recognize the same person even if they disappear and reappear, or appear in a different camera's feed.
  * **Database Integration**: Connects to a **MongoDB** database to store feature vectors of known individuals. This allows the system to identify registered people by name and role.
  * **Real-time Alerts**:
      * Triggers an audible beep, a desktop pop-up notification (`tkinter`), and saves a snapshot if an **unknown person** remains in view for a configurable duration.
      * Includes a cooldown mechanism to prevent alert spam.
  * **Historical Logging**: Logs every time a *known* person is identified, storing the person's name, camera, timestamp, and a thumbnail in a MongoDB collection for auditing.
  * **Centralized Dashboard**: Displays all camera feeds in a single, dynamically sized grid view using OpenCV for easy monitoring.

## 🛠️ How It Works (System Architecture)

The pipeline for each camera stream operates as follows:

1.  **Frame Capture**: The system reads a frame from the video source.
2.  **Detection**: The frame is passed to the **YOLO** model, which returns bounding boxes for all detected people.
3.  **Tracking**: The YOLO detections are fed into the **DeepSORT** tracker. DeepSORT manages object tracks, assigning a unique `track_id` to each person and predicting their movement to maintain consistency between frames.
4.  **Re-Identification**: For each tracked person, the system periodically:
      * Crops the person from the frame using their bounding box.
      * Passes the crop to the **OSNet (Re-ID) model** to generate a feature vector.
      * Compares this vector against the pre-enrolled feature vectors in the **MongoDB database** using cosine similarity.
5.  **Identification & Logic**:
      * If the similarity score passes a strict threshold, the person is identified with their registered name.
      * If the person cannot be matched, they are labeled as "UNKNOWN".
      * A voting and confidence mechanism is used to stabilize identity assignments and prevent flickering.
6.  **Alerting & Logging**:
      * If a track remains "UNKNOWN" for more than `ALERT_UNKNOWN_SECONDS`, an alert is triggered.
      * If a known person is identified, the event is logged to the `track_history` collection in MongoDB.
7.  **Visualization**: The processed frame, with bounding boxes and identity labels, is sent to the main dashboard for display.

## ⚙️ Tech Stack

  * **AI / ML**:
      * `PyTorch`: Core deep learning framework.
      * `ultralytics`: For the YOLO object detection model.
      * `torchreid`: For the OSNet person re-identification model.
      * `deep-sort-realtime`: For real-time object tracking.
  * **Database**:
      * `MongoDB`: Stores registered person data and historical track logs.
      * `pymongo`: Python driver for MongoDB.
  * **Core & Utility**:
      * `OpenCV-Python`: For all video and image processing tasks.
      * `NumPy` & `SciPy`: For numerical operations and calculating cosine distance.
      * `tkinter`: For creating native GUI alert pop-ups.

-----

## 🏁 Getting Started

### Prerequisites

  * Python 3.8+
  * MongoDB installed and running on `localhost:27017`.
  * An NVIDIA GPU with CUDA installed is **highly recommended** for real-time performance. The code will fall back to CPU if CUDA is not available.

### 1\. Clone the Repository

```bash
git clone https://github.com/adityapatil37/PIDS.git
cd PIDS
```

### 2\. Create a Virtual Environment & Install Dependencies

It's best practice to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deep-sort-realtime torchreid opencv-python pymongo scipy
```

> **Note**: The first command installs PyTorch with CUDA 11.8 support. Adjust the `cuXXX` version based on your CUDA installation or remove it to install the CPU version.

### 3\. Set Up the MongoDB Database

You need to manually create the database and collections and enroll at least one person for the system to identify.

1.  **Database Name**: `person_reid`

2.  **Collections**:

      * `people`: Stores information about known individuals.
      * `track_history`: Will be created automatically to log events.

3.  **Enroll a Person**:
    Insert a document into the `people` collection with a name and role. The `features` array will be populated by a separate enrollment script (not included), but you can create the document structure first.

    **Example `people` document:**

    ```json
    {
      "name": "John Doe",
      "role": "Employee",
      "features": [
        // This array will contain the 512-dimensional feature vectors
        // generated from multiple images of John Doe during an
        // enrollment process.
      ]
    }
    ```

    > **Important**: The system will only identify people if their feature vectors are stored in this collection. Without an enrollment script, the system will label everyone as "UNKNOWN".

### 4\. Configure the Application

Open `app.py` and modify the following sections as needed:

  * **Camera Sources**: Update the `camera_list` at the bottom of the script with your video file paths or camera indices.

    ```python
    if __name__ == "__main__":
        camera_list = [
            ("path/to/your/video1.mp4", "Lobby Cam"),
            ("path/to/your/video2.mp4", "Entrance Cam"),
            # (0, "Webcam 1") # Use 0 for the default webcam
        ]
        # ...
    ```

  * **Tuning Parameters**: Adjust the matching and alert thresholds at the top of the file for your specific environment.

    ```python
    # Matching params
    STRICT_TH = 0.35   # Lower for more strict matching
    LOOSE_TH = 0.55    # The outer bound for a potential match
    YOLO_CONF_TH = 0.45 # Minimum confidence for a person detection

    # Alerting params
    ALERT_UNKNOWN_SECONDS = 5  # Time an unknown person must be visible to trigger an alert
    ALERT_COOLDOWN = 10        # Seconds between alerts for the same camera
    ```

### 5\. Run the System

Execute the script from your terminal:

```bash
python app.py
```

The application will start, process the camera feeds, and open a dashboard window. Press **'q'** with the dashboard window in focus to quit the application gracefully.

-----

## 📁 Project Structure

```
.
├── app.py                  # Main application script
├── alert_snapshots/        # Directory created automatically to save alert images
└── unknown_crops/          # Directory created automatically for unknown person thumbnails
```

## 💡 Future Enhancements

  * **Web-Based UI**: Replace the OpenCV/Tkinter UI with a web interface (e.g., using Flask/FastAPI and WebSockets) for remote access and a better user experience.
  * **Enrollment Script**: Create a utility script to easily enroll new people by processing a folder of their images to generate and save feature vectors to MongoDB.
  * **Advanced Alerting**: Integrate with services like Twilio (SMS), SendGrid (email), or Slack/Discord for more robust notifications.
  * **Cross-Camera Tracking**: Implement more advanced logic to see if a `track_id` that disappears from one camera reappears on another shortly after, enabling true cross-camera tracking.
  * **Performance Optimization**: Use TensorRT to optimize the models for higher FPS on NVIDIA GPUs.