# backend/main.py
import cv2
import mediapipe as mp
import math
import io
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, Float, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from PIL import Image

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
mp_pose = mp.solutions.pose
mp_drawings = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    rads = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(rads * 180 / math.pi)
    return 360 - angle if angle > 180 else angle

# ================== API Routes ==================
@app.post("/analyze_frame")
async def analyze_frame(
    file: UploadFile,
    exercise: str = Form("squat"),
    user_id: int = Form(1)
):
    # Read image from request
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    response = {"exercise": exercise, "angle": None, "form": "good", "stage": None, "reps": 0}

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        if exercise == "squat":
            hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            angle = calculate_angle(hip, knee, ankle)

            response["angle"] = round(angle, 2)
            if angle > 160:
                response["stage"] = "up"
            elif angle < 90:
                response["stage"] = "down"
                response["reps"] += 1
            if angle < 70 or angle > 170:
                response["form"] = "bad"

    # Save session summary
    db = SessionLocal()
    session = Session(user_id=user_id, exercise=exercise, reps=response["reps"])
    db.add(session)
    db.commit()
    db.refresh(session)
    db.close()

    return response

@app.post("/start_session")
def start_session(user_id: int = Form(...), exercise: str = Form(...)):
    db = SessionLocal()
    session = Session(user_id=user_id, exercise=exercise, reps=0)
    db.add(session)
    db.commit()
    db.refresh(session)
    db.close()
    return {"message": "Session started", "session_id": session.id}

@app.get("/history/{user_id}")
def history(user_id: int):
    db = SessionLocal()
    sessions = db.query(Session).filter(Session.user_id == user_id).all()
    db.close()
    return [{"exercise": s.exercise, "reps": s.reps} for s in sessions]
