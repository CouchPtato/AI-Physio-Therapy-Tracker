from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from PIL import Image
import numpy as np
import cv2
import os
import sys

# Add the exercise_classifier folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../exercise_classifier")))
from exercise_classifier.inference import classify_exercise  # Import the classifier function

# ================== DB Setup ==================
Base = declarative_base()
engine = create_engine("sqlite:///exercise.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise = Column(String)
    reps = Column(Integer)
    user = relationship("User", back_populates="sessions")

Base.metadata.create_all(bind=engine)

# ================== FastAPI Setup ==================
app = FastAPI()

# Allow CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Mediapipe Setup ==================
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawings = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    rads = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(rads * 180 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# ================== API Routes ==================
@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile, exercise: str = Form(...), user_id: str = Form(...)):
    # Save the uploaded file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read the image using OpenCV
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        os.remove(file_path)  # Clean up the temporary file
        return {"error": "No pose landmarks detected"}

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # Calculate the knee angle
    angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Classify the exercise quality using the classifier
    classification = classify_exercise(file_path)  # Call the classifier from inference.py

    # Clean up the temporary file
    os.remove(file_path)

    # Return the analysis results
    return {
        "exercise": exercise,
        "angle": angle,
        "form": classification,  # Good, Partial, or Bad
        "user_id": user_id,
    }

@app.post("/start_session")
async def start_session(user_id: str = Form(...), exercise: str = Form(...)):
    db = SessionLocal()
    session = Session(user_id=user_id, exercise=exercise, reps=0)
    db.add(session)
    db.commit()
    db.refresh(session)
    db.close()
    return {"session_id": session.id}

@app.get("/history/{user_id}")
async def history(user_id: int):
    db = SessionLocal()
    sessions = db.query(Session).filter(Session.user_id == user_id).all()
    db.close()
    return [{"exercise": s.exercise, "reps": s.reps} for s in sessions]