# extract_features.py
import glob, os, numpy as np
from collect_data import process_video_capture, extract_landmarks_from_frame, FEATURE_DIM, NUM_LANDMARKS
import cv2
import mediapipe as mp
from tqdm import tqdm

mp_pose = mp.solutions.pose

def process_video_file(video_path, seq_len=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            frames.append(extract_landmarks_from_frame(results))
    cap.release()
    if len(frames) == 0:
        return []
    frames = np.stack(frames)
    seqs = []
    stride = seq_len // 2
    start = 0
    while start < len(frames):
        end = start + seq_len
        seq = frames[start:end]
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), FEATURE_DIM), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        seqs.append(seq)
        start += stride
    return seqs

if __name__ == "__main__":
    raw_dir = "data/raw_videos"
    out_dir = "data/sequences"
    os.makedirs(out_dir, exist_ok=True)
    seq_len = 60
    for vid in sorted(glob.glob(os.path.join(raw_dir, "*.mp4"))):
        # expecting filenames like pushup_01.mp4 or squat_02.mp4
        base = os.path.basename(vid)
        label = base.split("_")[0]
        seqs = process_video_file(vid, seq_len=seq_len)
        for i, s in enumerate(seqs):
            fname = f"{label}_{os.path.splitext(base)[0]}_{i:03d}.npy"
            np.save(os.path.join(out_dir, fname), s)
        print(f"Processed {vid} -> {len(seqs)} sequences")
