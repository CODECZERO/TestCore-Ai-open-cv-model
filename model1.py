import cv2
import mediapipe as mp
import csv
import numpy as np
from scipy.spatial import distance as dist
import time
from tensorflow.keras.models import load_model
import tensorflow as tf

# Use CPU (disable GPU)
tf.config.set_visible_devices([], 'GPU')

# ================================
# MediaPipe Setup
# ================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define Eye Landmark Indices for MediaPipe Face Mesh (for one face)
MP_LEFT_EYE = [362, 385, 387, 263, 373, 380]   # Left eye landmarks
MP_RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # Right eye landmarks

# Landmark indices for vertical measurement (approximate head tilt)
VERTICAL_TOP = 10      # e.g. near the forehead
VERTICAL_BOTTOM = 152  # e.g. near the chin

# ================================
# Helper Function: Compute EAR
# ================================
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])  # vertical distance
    B = dist.euclidean(eye_points[2], eye_points[4])  # vertical distance
    C = dist.euclidean(eye_points[0], eye_points[3])  # horizontal distance
    return (A + B) / (2.0 * C)

# ================================
# CSV Logging Setup
# ================================
csv_file = open('eye_tracking_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "LeftEAR", "RightEAR", "VerticalDiff", "Label"])

# ================================
# Video Capture Setup
# ================================
cap = cv2.VideoCapture(0)

# ================================
# Calibration and Baseline Update Parameters
# ================================
calibration_duration = 10   # seconds for initial calibration
alpha = 0.05                # EMA smoothing factor (new data weight)
calibration_start = time.time()
baseline_left = None
baseline_right = None
baseline_vertical = None

# ================================
# Alert Control and Event Counters
# ================================
alert_end_time = 0          # time until which alert is active
THRESHOLD = 0.05            # EAR deviation threshold
THRESHOLD_v = 0.02          # Vertical difference deviation threshold

center_count = 0
cheating_count = 0
no_face_count = 0
glasses_count = 0          # Counter for detected glasses (heuristic)
total_frames = 0

# ================================
# Load Pretrained Eye Tracking Model (Keras format)
# ================================
model = load_model('eye_tracking_model_final.keras')  # Ensure this file exists

# ================================
# Main Loop: Process Video Frames
# ================================
with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, 
                            min_tracking_confidence=0.7) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        current_time = time.time()
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        left_ear = right_ear = vertical_diff = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Extract eye landmarks and convert to pixel coordinates
            left_eye_coords = np.array([(face_landmarks.landmark[i].x, 
                                          face_landmarks.landmark[i].y) for i in MP_LEFT_EYE])
            right_eye_coords = np.array([(face_landmarks.landmark[i].x, 
                                           face_landmarks.landmark[i].y) for i in MP_RIGHT_EYE])
            left_eye_coords = np.multiply(left_eye_coords, [w, h]).astype(int)
            right_eye_coords = np.multiply(right_eye_coords, [w, h]).astype(int)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_coords)
            right_ear = eye_aspect_ratio(right_eye_coords)

            # Compute vertical difference (using normalized coordinates)
            top_point = face_landmarks.landmark[VERTICAL_TOP]
            bottom_point = face_landmarks.landmark[VERTICAL_BOTTOM]
            vertical_diff = bottom_point.y - top_point.y

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # After calibration_duration, update baseline continuously using EMA
            if current_time - calibration_start >= calibration_duration:
                if baseline_left is None:
                    # Initialize baseline on first post-calibration frame
                    baseline_left = left_ear
                    baseline_right = right_ear
                    baseline_vertical = vertical_diff
                    print("Initial Baseline computed: Left =", baseline_left, 
                          "Right =", baseline_right, "Vertical =", baseline_vertical)
                else:
                    # Update baseline using exponential moving average
                    baseline_left = alpha * left_ear + (1 - alpha) * baseline_left
                    baseline_right = alpha * right_ear + (1 - alpha) * baseline_right
                    baseline_vertical = alpha * vertical_diff + (1 - alpha) * baseline_vertical

        else:
            no_face_count += 1
            cv2.putText(frame, "⚠ No Face Detected!", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Log and determine state if measurements are available
        if left_ear is not None and right_ear is not None and vertical_diff is not None:
            # Log the raw measurements (timestamp and values)
            csv_writer.writerow([current_time, left_ear, right_ear, vertical_diff, ""])
            
            # Prepare input features for the model: [left_ear, right_ear, vertical_diff]
            input_data = np.array([[left_ear, right_ear, vertical_diff]])
            input_data = input_data.reshape((1, 3))  # Model expects shape (1,3)
            prediction = model.predict(input_data)
            # For binary classification, assume the model outputs a probability for "Cheating"
            ml_prob = prediction[0][0]

            alert_triggered = False
            direction = "Looking Center"
            
            # Rule-based logic: Compare current EAR and vertical diff with baseline
            if baseline_left is not None and baseline_right is not None and baseline_vertical is not None:
                if left_ear < baseline_left - THRESHOLD:
                    direction = "Looking Left"
                    alert_triggered = True
                elif right_ear < baseline_right - THRESHOLD:
                    direction = "Looking Right"
                    alert_triggered = True

                if vertical_diff > baseline_vertical + THRESHOLD_v:
                    direction = "Looking Down"
                    alert_triggered = True
                elif vertical_diff < baseline_vertical - THRESHOLD_v:
                    direction = "Looking Up"
                    alert_triggered = True

            # If the ML model predicts cheating (probability > 0.5), override the decision
            if ml_prob > 0.5:
                direction = "Cheating!"
                alert_triggered = True

            # Update alert duration: if triggered, set alert to last 30 seconds from now
            if alert_triggered and current_time > alert_end_time:
                alert_end_time = current_time + 30

            # Update counters
            if current_time < alert_end_time and alert_triggered:
                cheating_count += 1
            else:
                center_count += 1

            # Log decision in CSV
            csv_writer.writerow([current_time, left_ear, right_ear, vertical_diff, direction])
        else:
            center_count += 1

        # Create a side panel to display statistics (300px wide)
        side_panel = np.full((h, 300, 3), 255, dtype=np.uint8)  # white background
        cv2.putText(side_panel, "Stats:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(side_panel, f"Center: {center_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0), 2)
        cv2.putText(side_panel, f"Cheating: {cheating_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(side_panel, f"No Face: {no_face_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(side_panel, f"Glasses: {glasses_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cheat_percentage = (cheating_count / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(side_panel, f"Cheat %: {cheat_percentage:.1f}%", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Draw an alert message on the main frame if alert is active
        if current_time < alert_end_time:
            cv2.putText(frame, direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Looking Center", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        # Combine the main frame and side panel horizontally and show the result
        combined_frame = np.hstack((frame, side_panel))
        cv2.imshow('Eye Tracking - Stats Panel', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup resources
cap.release()
csv_file.close()
cv2.destroyAllWindows()

