# collect_data.py
import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm

mp_pose = mp.solutions.pose

# Landmarks per frame: 33 landmarks x (x,y,z,visibility) = 132 features
NUM_LANDMARKS = 33
FEATURES_PER_LM = 4
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LM

def extract_landmarks_from_frame(results):
    # returns flattened vector of length FEATURE_DIM
    if results.pose_landmarks is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    data = []
    for lm in results.pose_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(data, dtype=np.float32)

def process_video_capture(label, out_dir, seq_len=60, video_path=None, use_webcam=False):
    os.makedirs(out_dir, exist_ok=True)
    cap = None
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/raw_videos/squat_01.mp4')
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frames = []
        saved = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_webcam else None)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            lm_vec = extract_landmarks_from_frame(results)
            frames.append(lm_vec)
            pbar.update(1) if pbar.total else None

            # Optionally show
            cv2.putText(frame, f"Recording {label} - frames {len(frames)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        pbar.close()
        cv2.destroyAllWindows()

        # sliding windows -> save .npy sequences
        if len(frames) == 0:
            print("No frames captured.")
            return
        frames = np.stack(frames)  # (T, FEATURE_DIM)
        # make sequences: sliding window stride = seq_len (non-overlapping) OR overlap if you prefer
        stride = seq_len // 2
        start = 0
        idx = 0
        while start < len(frames):
            end = start + seq_len
            seq = frames[start:end]
            if len(seq) < seq_len:
                # pad with zeros (or repeat last frame)
                pad = np.zeros((seq_len - len(seq), FEATURE_DIM), dtype=np.float32)
                seq = np.concatenate([seq, pad], axis=0)
            fname = f"{label}_{idx:04d}.npy"
            np.save(os.path.join(out_dir, fname), seq)
            idx += 1
            start += stride
        print(f"Saved {idx} sequences for label {label} into {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="exercise label, e.g., 'pushup'")
    parser.add_argument("--out", default="data/sequences", help="output folder for sequences")
    parser.add_argument("--seq_len", type=int, default=60, help="sequence length in frames")
    parser.add_argument("--video", default=None, help="path to video file (if not using webcam)")
    parser.add_argument("--webcam", action="store_true", help="use webcam instead of file")
    args = parser.parse_args()
    process_video_capture(args.label, args.out, seq_len=args.seq_len, video_path=args.video, use_webcam=args.webcam)
