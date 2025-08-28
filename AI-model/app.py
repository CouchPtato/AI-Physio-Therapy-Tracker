import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

exercises = {
    "Squat": {"joints": ("hip", "knee", "ankle"), "good_range": (70, 100)},
    "Lunge": {"joints": ("hip", "knee", "ankle"), "good_range": (80, 110)},
    "Shoulder Raise": {"joints": ("shoulder", "elbow", "wrist"), "good_range": (70, 110)},
    "Bicep Curl": {"joints": ("shoulder", "elbow", "wrist"), "good_range": (30, 60)}
}

st.title("AI Therapy Exercise Tracker 🏋️‍♂️")

exercise_choice = st.selectbox("Choose Exercise", list(exercises.keys()))

# -------------------------------
# Capture video
# -------------------------------
cap = cv2.VideoCapture(0)
counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            ex = exercises[exercise_choice]
            joint_names = ex["joints"]
            good_range = ex["good_range"]
            
            lm = {
                "hip": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                "knee": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                "ankle": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                "shoulder": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                "elbow": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                "wrist": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            }
            
            angle = calculate_angle(
                (lm[joint_names[0]].x, lm[joint_names[0]].y),
                (lm[joint_names[1]].x, lm[joint_names[1]].y),
                (lm[joint_names[2]].x, lm[joint_names[2]].y)
            )
            
            if angle > good_range[1]:
                stage = "down"
            if angle < good_range[0] and stage == "down":
                stage = "up"
                counter += 1
            
            cv2.putText(image, f'Exercise: {exercise_choice}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Angle: {int(angle)}', (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Reps: {counter}', (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f'State: {stage}', (10,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            
        except:
            pass
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('AI Therapy', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
