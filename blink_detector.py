import cv2
import numpy as np
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import mediapipe as mp
import os
import time

# MediaPipe Face Landmarker setup (new API)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download model if needed
import urllib.request
import os

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

detector = FaceLandmarker.create_from_options(options)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink detection parameters
EAR_THRESHOLD = 0.25
BLINK_COUNTER = 0
EYES_CLOSED = False

# BPM tracking (rolling 60-second window)
blink_timestamps = deque()
bpm_history = deque(maxlen=100)

# Sound alert tracking (prevent continuous beeping)
last_alert_time = 0
ALERT_COOLDOWN = 3  # seconds between alerts

# Matplotlib setup for live graph
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlabel('Time')
ax.set_ylabel('Blinks Per Minute')
ax.set_title('Live Blink Rate Monitor')
ax.set_ylim(0, 30)
line, = ax.plot([], [], 'b-', linewidth=2)
threshold_line = ax.axhline(y=20, color='r', linestyle='--', label='Focus Drift Threshold')
ax.legend()
ax.grid(True)


def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR)"""
    coords = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
    
    # Vertical distances
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    
    # Horizontal distance
    h = np.linalg.norm(coords[0] - coords[3])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear


def calculate_bpm():
    """Calculate blinks per minute in rolling 60-second window"""
    current_time = datetime.now()
    
    # Remove blinks older than 60 seconds
    while blink_timestamps and (current_time - blink_timestamps[0]).total_seconds() > 60:
        blink_timestamps.popleft()
    
    return len(blink_timestamps)


def update_graph():
    """Update the BPM graph"""
    if len(bpm_history) > 0:
        line.set_data(range(len(bpm_history)), list(bpm_history))
        ax.set_xlim(0, max(100, len(bpm_history)))
    fig.canvas.draw()
    fig.canvas.flush_events()


def play_alert_sound():
    """Play alert sound (macOS compatible)"""
    try:
        # macOS system sound
        os.system('afplay /System/Library/Sounds/Ping.aiff &')
    except:
        # Fallback: print to terminal
        print('\a')  # Terminal bell


# Start webcam
cap = cv2.VideoCapture(0)
frame_counter = 0

print("Eye Blink Detector Started")
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process with MediaPipe (use frame counter as timestamp)
    timestamp_ms = frame_counter
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink detection logic
        if avg_ear < EAR_THRESHOLD:
            BLINK_COUNTER += 1
            EYES_CLOSED = True
        else:
            if EYES_CLOSED and BLINK_COUNTER >= 1:  # Changed from 2 to 1 for fast blinks
                # Blink detected
                blink_timestamps.append(datetime.now())
                EYES_CLOSED = False
            BLINK_COUNTER = 0
        
        # Calculate BPM
        bpm = calculate_bpm()
        bpm_history.append(bpm)
        
        # Display metrics on frame
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"BPM: {bpm}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Blinks: {len(blink_timestamps)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Focus drift detection
        if bpm > 20:
            cv2.putText(frame, "FOCUS DRIFT DETECTED", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Play alert sound (with cooldown)
            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                play_alert_sound()
                last_alert_time = current_time
        
        # Draw eye landmarks
        h, w = frame.shape[:2]
        for idx in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
    
    # Update graph periodically
    if frame_counter % 5 == 0:
        update_graph()
    
    # Display video feed
    cv2.imshow('Eye Blink Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close()
detector.close() 