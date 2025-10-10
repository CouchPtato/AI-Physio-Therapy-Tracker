import cv2
import mediapipe as mp
import csv
import math
import os

mp_pose = mp.solutions.pose
mp_drawings = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

def calculate_angle(a, b, c):

    rads = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(rads * 180 / math.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

exercise_name = "bicep_curl"   # "bicep_curl", "squat", "shoulder_abduction"

camera = cv2.VideoCapture(0)

csv_file = "exercise_labeled.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode = "a", newline = "") as f:
    csv_writer = csv.writer(f)

    if not file_exists:
        header = ["exercise", "frame", "angle", "count", "stage", "form"]
        csv_writer.writerow(header)

    frame_count = 0
    counter = 0
    stage = None

    while camera.isOpened():
        status, frame = camera.read()
        if not status:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            form = "good"
            angle = None

            if exercise_name == "bicep_curl":
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                if angle < 50 and stage == "down":
                    stage = "up"
                    counter += 1
                if angle < 30 or angle > 170:
                    form = "bad"

            elif exercise_name == "squat":
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1
                if angle < 70 or angle > 170:
                    form = "bad"

            elif exercise_name == "shoulder_abduction":
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                angle = calculate_angle(hip, shoulder, elbow)

                if angle < 30:
                    stage = "down"
                if angle > 80 and stage == "down":  # arm lifted to side
                    stage = "up"
                    counter += 1
                if angle < 20 or angle > 120:
                    form = "bad"

            # Save to CSV
            if angle is not None:
                row = [exercise_name, frame_count, angle, counter, stage, form]
                csv_writer.writerow(row)

                # Draw pose + info
                mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(image, f'Angle: {int(angle)}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Reps: {counter}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Stage: {stage}', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Form: {form}', (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            frame_count += 1

        cv2.imshow('Exercise Tracker - Press Q to stop', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()