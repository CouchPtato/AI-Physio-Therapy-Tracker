import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import time
import numpy as np
from datetime import datetime
from fpdf import FPDF

st.set_page_config(page_title="AI Physio Tracker", layout="wide")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ========================
# Exercise definitions
# ========================
EXERCISES = {
    "bicep_curl": ["LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST"],
    "squat": ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"],
    "shoulder_abduction": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"],
    "knee_extension": ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"],
    "leg_raise": ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"],
    "side_bend": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
}

# ========================
# Helper functions
# ========================
def draw_skeleton(image, landmarks, involved_names):
    h, w = image.shape[:2]
    involved_indices = [
        mp_pose.PoseLandmark[name].value
        for name in involved_names
        if name in mp_pose.PoseLandmark.__members__
    ]
    for conn in mp_pose.POSE_CONNECTIONS:
        a_idx = conn[0].value if hasattr(conn[0], "value") else conn[0]
        b_idx = conn[1].value if hasattr(conn[1], "value") else conn[1]
        if a_idx in involved_indices and b_idx in involved_indices:
            a_point = landmarks[a_idx]
            b_point = landmarks[b_idx]
            if a_point.visibility > 0.5 and b_point.visibility > 0.5:
                cv2.line(
                    image,
                    (int(a_point.x * w), int(a_point.y * h)),
                    (int(b_point.x * w), int(b_point.y * h)),
                    (0, 255, 0),
                    2,
                )
    return image


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


# ========================
# Enhanced form scoring
# ========================
def assess_form(exercise, angle):
    """Enhanced dynamic score + same feedback messages."""
    ideal_ranges = {
        "bicep_curl": (30, 160),
        "squat": (70, 160),
        "shoulder_abduction": (70, 160),
        "knee_extension": (0, 160),
        "leg_raise": (40, 150),
        "side_bend": (10, 35),
    }
    ideal_min, ideal_max = ideal_ranges.get(exercise, (60, 150))

    # Dynamic continuous score
    if angle < ideal_min:
        diff = ideal_min - angle
    elif angle > ideal_max:
        diff = angle - ideal_max
    else:
        diff = 0
    score = max(0.0, 1.0 - (diff / 60))

    # Keep your existing feedback
    feedback = "Keep form consistent."
    if exercise == "bicep_curl":
        if angle < 60:
            feedback = "Great contraction!"
        elif angle > 160:
            feedback = "Full extension!"
        else:
            feedback = "Complete your motion fully."
    elif exercise == "squat":
        if angle < 95:
            feedback = "Nice deep squat!"
        elif angle > 160:
            feedback = "Standing tall."
        else:
            feedback = "Try going a bit lower."
    elif exercise == "shoulder_abduction":
        feedback = "Good arm raise!" if angle > 120 else "Lift higher for full range."
    elif exercise == "knee_extension":
        feedback = "Full knee extension achieved!" if angle > 160 else "Straighten knee more."
    elif exercise == "leg_raise":
        feedback = "Leg raised high enough!" if angle > 140 else "Lift leg higher."
    elif exercise == "side_bend":
        feedback = "Nice side bend!" if 15 < angle < 35 else "Bend slightly more to side."

    return feedback, score


def generate_pdf_report(exercise, reps, duration, avg_speed, avg_score, feedback_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, txt="Physiotherapy Session Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Exercise: {exercise}", ln=True)
    pdf.cell(200, 10, txt=f"Total Reps: {reps}", ln=True)
    pdf.cell(200, 10, txt=f"Session Duration: {duration:.1f} seconds", ln=True)
    pdf.cell(200, 10, txt=f"Average Speed: {avg_speed:.2f} units/s", ln=True)
    pdf.cell(200, 10, txt=f"Form Score: {avg_score*100:.1f}/100", ln=True)
    pdf.multi_cell(200, 10, txt=f"Feedback Summary: {feedback_summary}")

    filename = f"{exercise}_report_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# ========================
# Streamlit UI
# ========================
st.title("🏋️ AI Physiotherapy Exercise Tracker")
st.sidebar.header("Exercise & Input Settings")

exercise = st.sidebar.selectbox("Select Exercise", list(EXERCISES.keys()))
mode = st.sidebar.radio("Choose Input Mode", ["Live Webcam", "Upload Video"])

for key in ["running", "reps", "rep_times", "last_rep_ts", "stage", "form_scores", "feedbacks"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["rep_times", "form_scores", "feedbacks"] else None if key in ["stage", "last_rep_ts"] else 0 if key == "reps" else False

frame_placeholder = st.empty()
uploaded_video = None
if mode == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload Exercise Video", type=["mp4", "mov", "avi"])

start_btn = st.sidebar.button("▶️ Start Session")
end_btn = st.sidebar.button("🛑 End Session")

# ========================
# Video Capture Logic
# ========================
if start_btn:
    st.session_state.running = True
    st.session_state.reps = 0
    st.session_state.rep_times = []
    st.session_state.last_rep_ts = None
    st.session_state.stage = None
    st.session_state.start_time = time.time()
    st.session_state.form_scores = []
    st.session_state.feedbacks = []

if end_btn:
    st.session_state.running = False
    duration = time.time() - st.session_state.start_time
    avg_speed = np.mean(st.session_state.rep_times) if st.session_state.rep_times else 0
    avg_score = np.mean(st.session_state.form_scores) if st.session_state.form_scores else 0
    feedback_summary = ", ".join(list(set(st.session_state.feedbacks))) or "Maintain consistent posture."

    report_file = generate_pdf_report(exercise, st.session_state.reps, duration, avg_speed, avg_score, feedback_summary)
    st.success("✅ Session Ended. Report Generated Below.")
    st.write(f"**Exercise:** {exercise}")
    st.write(f"**Reps:** {st.session_state.reps}")
    st.write(f"**Duration:** {duration:.1f} sec")
    st.write(f"**Average Speed:** {avg_speed:.2f}")
    st.write(f"**Form Score:** {avg_score*100:.1f}/100")
    st.write(f"**Feedback Summary:** {feedback_summary}")

    with open(report_file, "rb") as f:
        st.download_button("📄 Download Report", f, file_name=report_file)
    st.stop()

if st.session_state.running:
    if mode == "Live Webcam":
        cap = cv2.VideoCapture(0)
        time.sleep(1)
    else:
        if uploaded_video is None:
            st.warning("Please upload a video before starting.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Could not open video source.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / fps
    prev_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            display = frame.copy()

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                involved = EXERCISES[exercise]
                display = draw_skeleton(display, lm, involved)

                # ===== Compute angle =====
                angles = []
                if exercise == "bicep_curl":
                    for side in ["LEFT", "RIGHT"]:
                        shoulder = [lm[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
                                    lm[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y]
                        elbow = [lm[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
                                 lm[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y]
                        wrist = [lm[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
                                 lm[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y]
                        angles.append(calculate_angle(shoulder, elbow, wrist))
                elif exercise == "squat":
                    for side in ["LEFT", "RIGHT"]:
                        hip = [lm[mp_pose.PoseLandmark[f"{side}_HIP"].value].x,
                               lm[mp_pose.PoseLandmark[f"{side}_HIP"].value].y]
                        knee = [lm[mp_pose.PoseLandmark[f"{side}_KNEE"].value].x,
                                lm[mp_pose.PoseLandmark[f"{side}_KNEE"].value].y]
                        ankle = [lm[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].x,
                                 lm[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].y]
                        angles.append(calculate_angle(hip, knee, ankle))
                else:
                    angles = [0.0]

                angle = np.mean(angles)
                feedback, score = assess_form(exercise, angle)
                st.session_state.form_scores.append(score)
                st.session_state.feedbacks.append(feedback)

                # ===== Rep detection =====
                down_thresh, up_thresh = 100, 160
                if angle < down_thresh:
                    st.session_state.stage = "down"
                elif angle > up_thresh and st.session_state.stage == "down":
                    st.session_state.reps += 1
                    now = time.time()
                    if st.session_state.last_rep_ts:
                        rep_time = now - st.session_state.last_rep_ts
                        st.session_state.rep_times.append(rep_time)
                    st.session_state.last_rep_ts = now
                    st.session_state.stage = "up"

                # ===== Display metrics =====
                live_form_score = np.mean(st.session_state.form_scores[-10:]) * 100 if st.session_state.form_scores else 0
                cv2.putText(display, f"{exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"Reps: {st.session_state.reps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"Angle: {int(angle)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"Form: {feedback}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(display, f"Form Score: {live_form_score:.1f}%", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,215,0), 2)
            else:
                cv2.putText(display, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            frame_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            elapsed = time.time() - prev_time
            time.sleep(max(0, frame_interval - elapsed))
            prev_time = time.time()

    cap.release()
    st.session_state.running = False
    cv2.destroyAllWindows()
