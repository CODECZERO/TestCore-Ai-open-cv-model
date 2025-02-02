import cv2
import mediapipe as mp
import csv
import numpy as np
from scipy.spatial import distance as dist
import time
from tensorflow.keras.models import load_model

# ================================
# MediaPipe Setup
# ================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define Eye Landmark Indices for MediaPipe Face Mesh (for one face)
MP_LEFT_EYE = [362, 385, 387, 263, 373, 380]   # Left eye landmarks
MP_RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # Right eye landmarks

# Landmark indices for vertical measurement (for approximate head tilt)
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
update_interval = 30        # seconds to update baseline
calibration_duration = 10   # seconds for initial calibration
ear_samples = []            # list to store (left_ear, right_ear) samples
vertical_samples = []       # list to store vertical difference samples
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
model = load_model('eye_tracking_model_final.keras')  # Ensure this file exists in the new native format

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
            # Use the first detected face
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

            # Compute vertical difference using two landmarks (normalized coordinates)
            top_point = face_landmarks.landmark[VERTICAL_TOP]
            bottom_point = face_landmarks.landmark[VERTICAL_BOTTOM]
            vertical_diff = bottom_point.y - top_point.y

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        else:
            no_face_count += 1
            cv2.putText(frame, "⚠ No Face Detected!", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Log and calibrate if EAR values and vertical_diff are computed
        if left_ear is not None and right_ear is not None and vertical_diff is not None:
            csv_writer.writerow([current_time, left_ear, right_ear, vertical_diff, ""])
            
            # Collect calibration samples during the first calibration_duration seconds
            if current_time - calibration_start < calibration_duration:
                ear_samples.append((left_ear, right_ear))
                vertical_samples.append(vertical_diff)
            else:
                # Compute baseline if not computed yet
                if baseline_left is None or baseline_right is None or baseline_vertical is None:
                    if ear_samples and vertical_samples:
                        baseline_left = np.mean([s[0] for s in ear_samples])
                        baseline_right = np.mean([s[1] for s in ear_samples])
                        baseline_vertical = np.mean(vertical_samples)
                        print("Initial Baseline computed: Left =", baseline_left, 
                              "Right =", baseline_right, "Vertical =", baseline_vertical)
                    else:
                        baseline_left, baseline_right, baseline_vertical = 0.25, 0.25, 0.25

                # Optionally update baseline periodically
                if current_time - calibration_start >= update_interval:
                    if ear_samples and vertical_samples:
                        baseline_left = np.mean([s[0] for s in ear_samples])
                        baseline_right = np.mean([s[1] for s in ear_samples])
                        baseline_vertical = np.mean(vertical_samples)
                        print("Baseline updated: Left =", baseline_left, 
                              "Right =", baseline_right, "Vertical =", baseline_vertical)
                        ear_samples = []
                        vertical_samples = []
                        calibration_start = current_time

                # Prepare input features for prediction:
                # Features: [left_ear, right_ear, vertical_diff]
                input_data = np.array([[left_ear, right_ear, vertical_diff]])
                input_data = input_data.reshape((1, 3))  # Reshape to (1, 3) to fit the model's expected shape

                prediction = model.predict(input_data)
                # Assuming binary classification: output probability for "Cheating"
                ml_prob = prediction[0][0]

                alert_triggered = False
                direction = "Looking Center"
                
                # Rule-based: Check horizontal deviation for each eye separately
                if left_ear < baseline_left - THRESHOLD:
                    direction = "Looking Left"
                    alert_triggered = True
                elif right_ear < baseline_right - THRESHOLD:
                    direction = "Looking Right"
                    alert_triggered = True

                # Rule-based: Check vertical deviation (head tilt)
                if vertical_diff > baseline_vertical + THRESHOLD_v:
                    direction = "Looking Down"
                    alert_triggered = True
                elif vertical_diff < baseline_vertical - THRESHOLD_v:
                    direction = "Looking Up"
                    alert_triggered = True

                # If ML model predicts "cheating" (probability > 0.5), override decision
                if ml_prob > 0.5:
                    direction = "Cheating!"
                    alert_triggered = True

                # Update alert duration: if triggered, alert lasts for 30 seconds (reset if new trigger)
                if alert_triggered and current_time > alert_end_time:
                    alert_end_time = current_time + 30

                # Update event counters
                if current_time < alert_end_time and alert_triggered:
                    cheating_count += 1
                else:
                    center_count += 1

                # Log decision in CSV
                csv_writer.writerow([current_time, left_ear, right_ear, vertical_diff, direction])
        else:
            center_count += 1

        # Create side panel for statistics (300px wide)
        side_panel = np.full((h, 300, 3), 255, dtype=np.uint8)  # white background
        cv2.putText(side_panel, "Stats:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(side_panel, f"Center: {center_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0), 2)
        cv2.putText(side_panel, f"Cheating: {cheating_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(side_panel, f"No Face: {no_face_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(side_panel, f"Glasses: {glasses_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cheat_percentage = (cheating_count / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(side_panel, f"Cheat %: {cheat_percentage:.1f}%", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Create an alert message on main frame if alert is active
        if current_time < alert_end_time:
            cv2.putText(frame, direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Looking Center", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        # Combine main frame and side panel horizontally
        combined_frame = np.hstack((frame, side_panel))
        cv2.imshow('Eye Tracking - Stats Panel', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
