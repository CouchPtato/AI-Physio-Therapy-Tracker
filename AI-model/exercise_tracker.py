# physio_tracker_final.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

# ---------------- Config ----------------
st.set_page_config(page_title="Physio Tracker — Enhanced", layout="wide")
st.title("🏋️ AI Physiotherapy Exercise Tracker — Enhanced")

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------------- Helpers ----------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def limited_draw(image, landmarks, involved_names):
    """
    Draw only connections where both endpoints are in involved_names.
    landmarks: list of landmark objects (NormalizedLandmark)
    involved_names: set of strings like 'LEFT_HIP'
    """
    h, w = image.shape[:2]
    # draw points (optional small circles) and connections
    for conn in mp_pose.POSE_CONNECTIONS:
        a_name = conn[0].name
        b_name = conn[1].name
        if a_name in involved_names and b_name in involved_names:
            a_idx = conn[0].value
            b_idx = conn[1].value
            ax = int(landmarks[a_idx].x * w)
            ay = int(landmarks[a_idx].y * h)
            bx = int(landmarks[b_idx].x * w)
            by = int(landmarks[b_idx].y * h)
            cv2.line(image, (ax, ay), (bx, by), (0, 255, 0), 3)
            cv2.circle(image, (ax, ay), 4, (0, 0, 255), -1)
            cv2.circle(image, (bx, by), 4, (0, 0, 255), -1)
    return image

def generate_pdf(path_pdf, summary, angle_series):
    """Create PDF with summary and angle plot."""
    c = canvas.Canvas(path_pdf, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(80, 800, "Physiotherapy Session Report")
    c.setFont("Helvetica", 12)
    y = 760
    c.drawString(80, y, f"Exercise: {summary['exercise']}")
    y -= 18
    c.drawString(80, y, f"Date: {summary['timestamp']}")
    y -= 18
    c.drawString(80, y, f"Duration (s): {summary['duration']:.2f}")
    y -= 18
    c.drawString(80, y, f"Repetitions: {summary['reps']}")
    y -= 18
    c.drawString(80, y, f"Average angle: {summary['avg_angle']:.2f}°")
    y -= 18
    c.drawString(80, y, f"Std angle: {summary['std_angle']:.2f}°")
    y -= 18
    c.drawString(80, y, f"Min angle: {summary['min_angle']:.2f}°")
    y -= 18
    c.drawString(80, y, f"Max angle: {summary['max_angle']:.2f}°")
    y -= 18
    c.drawString(80, y, f"Avg rep time (s): {summary.get('avg_rep_time',0):.2f}")
    y -= 18
    c.drawString(80, y, f"Form score (0-100): {summary['form_score']:.1f}")
    y -= 24

    # tips
    c.setFont("Helvetica-Bold", 13)
    c.drawString(80, y, "Tips:")
    y -= 16
    c.setFont("Helvetica", 11)
    for tip in summary.get("tips", []):
        c.drawString(90, y, f"- {tip}")
        y -= 14

    # angle plot
    if angle_series:
        fig, ax = plt.subplots(figsize=(5,2))
        ax.plot(angle_series, color="tab:blue")
        ax.set_title("Angle vs Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        fig_path = os.path.join(REPORTS_DIR, "tmp_plot.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        c.drawImage(fig_path, 80, 120, width=440, height=160)

    c.showPage()
    c.save()

def save_report(summary, angle_series):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base = f"{summary['exercise'].replace(' ','_')}_{ts}"
    json_path = os.path.join(REPORTS_DIR, fname_base + ".json")
    pdf_path = os.path.join(REPORTS_DIR, fname_base + ".pdf")
    # write json
    with open(json_path, "w") as jf:
        json.dump({"summary": summary, "angles": angle_series}, jf, indent=2)
    # write pdf
    generate_pdf(pdf_path, summary, angle_series)
    return pdf_path, json_path

# ---------------- Exercises config (only relevant joints)
EXERCISES = {
    "Squats": {
        "names": {"LEFT_HIP","LEFT_KNEE","LEFT_ANKLE","RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"},
        "angle_points": ("LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"),
        "down": 100, "up": 160
    },
    "Bicep Curls": {
        "names": {"LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST","RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"},
        "angle_points": ("RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"),
        "down": 45, "up": 160
    },
    "Shoulder Press": {
        "names": {"RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST","LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"},
        "angle_points": ("RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"),
        "down": 90, "up": 160
    },
    "Lunges": {
        "names": {"LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"},
        "angle_points": ("LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"),
        "down": 100, "up": 160
    },
    "Side Bends": {
        "names": {"RIGHT_SHOULDER","RIGHT_HIP","RIGHT_KNEE","LEFT_SHOULDER","LEFT_HIP","LEFT_KNEE"},
        "angle_points": ("RIGHT_SHOULDER","RIGHT_HIP","RIGHT_KNEE"),
        "down": 150, "up": 170
    },
    "Arm Raises": {
        "names": {"RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST","LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"},
        "angle_points": ("RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"),
        "down": 70, "up": 150
    }
}

# ---------------- UI ----------------
col1, col2 = st.columns([2,1])
with col1:
    exercise = st.selectbox("Select exercise", list(EXERCISES.keys()))
    mode = st.radio("Input type", ["Webcam","Upload video"])
with col2:
    st.write("Reports saved in:", REPORTS_DIR)
    # list saved reports
    files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")], reverse=True)
    if files:
        latest_pdf = files[0]
        st.markdown(f"**Latest report:** {latest_pdf}")
        if st.button("Show latest report preview"):
            st.write(f"Preview: {os.path.join(REPORTS_DIR, latest_pdf)}")
            st.video(os.path.join(REPORTS_DIR, latest_pdf))  # streamlit can preview PDF? video used as placeholder
    else:
        st.info("No saved reports yet.")

st.markdown("---")

# controls
start = st.button("▶ Start Tracking")
end = st.button("🛑 End Session (generate report)")
view_reports = st.button("📂 View saved reports")

# state
if "running" not in st.session_state:
    st.session_state.running = False
if "angles" not in st.session_state:
    st.session_state.angles = []
if "rep_times" not in st.session_state:
    st.session_state.rep_times = []
if "reps" not in st.session_state:
    st.session_state.reps = 0
if "stage" not in st.session_state:
    st.session_state.stage = "up"
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_rep_ts" not in st.session_state:
    st.session_state.last_rep_ts = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

# video placeholder
blank = np.zeros((480,640,3), dtype=np.uint8)
frame_placeholder = st.empty()

# start tracking
if start and not st.session_state.running:
    st.session_state.running = True
    st.session_state.angles = []
    st.session_state.rep_times = []
    st.session_state.reps = 0
    st.session_state.stage = "up"
    st.session_state.start_time = time.time()
    st.session_state.last_rep_ts = None
    st.session_state.last_summary = None

# open capture
cap = None
if st.session_state.running:
    if mode == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        upload = st.file_uploader("Upload video (start first, then pick file)", type=["mp4","mov","avi"])
        if upload:
            t = tempfile.NamedTemporaryFile(delete=False)
            t.write(upload.read())
            cap = cv2.VideoCapture(t.name)

# processing loop (non-blocking UI style using while but safe)
if cap is not None and st.session_state.running:
    with mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55) as pose:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                # end if video finished
                break
            # prepare frame
            frame = cv2.resize(frame, (640,480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            display = frame.copy()
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # angle points config
                pnames = EXERCISES[exercise]["angle_points"]
                p1_idx = mp_pose.PoseLandmark[pnames[0]].value
                p2_idx = mp_pose.PoseLandmark[pnames[1]].value
                p3_idx = mp_pose.PoseLandmark[pnames[2]].value

                a = [lm[p1_idx].x, lm[p1_idx].y]
                b = [lm[p2_idx].x, lm[p2_idx].y]
                c = [lm[p3_idx].x, lm[p3_idx].y]
                angle = calculate_angle(a,b,c)
                st.session_state.angles.append(angle)

                # rep detection
                down_thresh = EXERCISES[exercise]["down"]
                up_thresh = EXERCISES[exercise]["up"]
                # determine flags
                if angle < down_thresh:
                    st.session_state.stage = "down"
                elif angle > up_thresh and st.session_state.stage == "down":
                    # rep completed
                    st.session_state.reps += 1
                    now = time.time()
                    if st.session_state.last_rep_ts:
                        st.session_state.rep_times.append(now - st.session_state.last_rep_ts)
                    st.session_state.last_rep_ts = now
                    st.session_state.stage = "up"

                # draw only involved joints
                involved = EXERCISES[exercise]["names"]
                display = limited_draw(display, lm, involved)

                # show metrics overlay
                cv2.putText(display, f"{exercise}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"Reps: {st.session_state.reps}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"Angle: {int(angle)}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"Stage: {st.session_state.stage}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                # hint to user
                cv2.putText(display, "No pose detected - adjust camera", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # show frame
            frame_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

            # small pause to let UI update
            time.sleep(0.02)

    # release capture when loop ends
    cap.release()
    st.session_state.running = False

# End and produce report
if end:
    # compute stats
    duration = (time.time() - st.session_state.start_time) if st.session_state.start_time else 0.0
    angles = st.session_state.angles or []
    if angles:
        avg_angle = float(np.mean(angles))
        std_angle = float(np.std(angles))
        min_angle = float(np.min(angles))
        max_angle = float(np.max(angles))
    else:
        avg_angle = std_angle = min_angle = max_angle = 0.0
    reps = int(st.session_state.reps)
    avg_rep_time = float(np.mean(st.session_state.rep_times)) if st.session_state.rep_times else 0.0
    rep_rate = (reps / duration * 60) if duration > 0 else 0.0

    # form score heuristic: prefer avg angle inside exercise-specific ideal range
    # define ideal ranges per exercise (simple)
    ideal_ranges = {
        "Squats": (90, 140),
        "Bicep Curls": (30, 60),
        "Shoulder Press": (60, 110),
        "Lunges": (80, 130),
        "Side Bends": (140, 175),
        "Arm Raises": (60, 130)
    }
    ideal_min, ideal_max = ideal_ranges.get(exercise,(0,360))
    # score 0-100 by how many angle samples are within ideal range
    if angles:
        in_range_pct = 100.0 * (np.sum((np.array(angles) >= ideal_min) & (np.array(angles) <= ideal_max)) / len(angles))
    else:
        in_range_pct = 0.0
    form_score = float(in_range_pct)

    # generate textual tips
    tips = []
    if avg_angle < ideal_min:
        tips.append("Complete fuller range of motion (increase range).")
    if avg_angle > ideal_max:
        tips.append("Avoid overextending; control your movement.")
    if std_angle > 20:
        tips.append("Try to make movements smoother and consistent.")
    if reps == 0:
        tips.append("No repetitions detected — ensure you perform full reps within camera view.")

    summary = {
        "exercise": exercise,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "duration": duration,
        "reps": reps,
        "avg_angle": avg_angle,
        "std_angle": std_angle,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "avg_rep_time": avg_rep_time,
        "rep_rate_per_min": rep_rate,
        "form_score": form_score,
        "tips": tips
    }

    # save report files
    pdf_path, json_path = save_report(summary, angles)
    st.success("Session ended — report generated and saved.")
    st.subheader("Session Summary")
    st.json(summary)

    # visualization
    if angles:
        fig, ax = plt.subplots()
        ax.plot(angles, color='tab:orange')
        ax.set_title("Angle progression during session")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        st.pyplot(fig)

    st.markdown("### Download / view saved report")
    with open(pdf_path, "rb") as f:
        st.download_button("⬇ Download PDF report", data=f, file_name=os.path.basename(pdf_path), mime="application/pdf")

    st.write("Saved JSON:", json_path)
    st.write("Saved PDF:", pdf_path)
    st.session_state.last_summary = summary

# View saved reports listing
if view_reports:
    st.subheader("Saved reports")
    pdfs = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")], reverse=True)
    for p in pdfs:
        colA, colB = st.columns([3,1])
        colA.write(p)
        fullp = os.path.join(REPORTS_DIR, p)
        if colB.button(f"Download {p}", key=p):
            with open(fullp, "rb") as f:
                st.download_button(f"Download {p}", f, file_name=p, mime="application/pdf")

# show last summary quick card
if st.session_state.last_summary:
    st.markdown("---")
    st.subheader("Last saved report summary")
    st.json(st.session_state.last_summary)
