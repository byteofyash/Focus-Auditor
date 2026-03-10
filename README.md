# Focus Auditor: Real-Time Cognitive Fatigue Monitor

A real-time computer vision application designed to detect cognitive fatigue and focus drift by monitoring blink rates. The system utilizes MediaPipe's Face Landmarker to extract precise facial coordinates and calculates the Eye Aspect Ratio (EAR) to detect blinks, triggering audio-visual alerts when a user's focus begins to wane.

---

## Key Features

- **Real-Time Landmark Detection:** Utilizes MediaPipe for high-performance, low-latency facial landmark extraction.  
- **Algorithmic Blink Detection:** Calculates the Eye Aspect Ratio (EAR) using Euclidean distances between specific eye landmarks to accurately identify closed eyes.  
- **Dynamic BPM Tracking:** Implements a double-ended queue (`deque`) to maintain a rolling 60-second window, calculating accurate Blinks Per Minute (BPM) while filtering out noise.  
- **Live Data Visualization:** Features a real-time Matplotlib graph that plots BPM history against a predefined focus drift threshold.  
- **Automated Alert System:** Triggers a non-intrusive audio and visual warning when the BPM exceeds the threshold, complete with a cooldown mechanism to prevent alert spam.

---

## Technical Architecture

The core of the detection relies on the **Eye Aspect Ratio (EAR)**. For every video frame, the system extracts six coordinates per eye. The algorithm computes the vertical distances between the upper and lower eyelids and divides them by the horizontal distance between the eye corners. When the EAR falls below a specific threshold (`0.25`), a blink is registered.

To prevent false positives from micro-blinks or sensor noise, the system logs the timestamp of each valid blink into a rolling queue, dropping any data older than 60 seconds to maintain an accurate, real-time BPM metric.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/focus-auditor.git
cd focus-auditor
```

### 2. (Optional but recommended) Create and activate a virtual environment

```bash
python -m venv venv
```

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

Note: The script will automatically download the required `MediaPipe face_landmarker.task` model file on its first run.

---

## Usage

Run the main Python script:

```bash
python main.py
```

- Ensure you are in a well-lit environment for optimal facial landmark detection.  
- Press **`q`** to quit the application safely and release the webcam resources.

---

## Future Scope

- Implement dynamic EAR thresholding calibrated to individual users during an initial baseline phase.
- Integrate head pose estimation to detect physical slumping or looking away from the screen.