# predict_video.py
import cv2, os, numpy as np, json
from tensorflow.keras.models import load_model
import mediapipe as mp
from collect_data import extract_landmarks_from_frame, FEATURE_DIM

SEQ_LEN = 60
MODEL_PATH = "models/best_model.h5"
LABELS_PATH = "models/labels_map.json"

mp_pose = mp.solutions.pose

def load_label_map(path):
    with open(path, "r") as f:
        d = json.load(f)
    # invert map: idx->label
    inv = {int(v): k for k, v in d.items()}
    return inv

def predict_on_video(video_path, model, labels_map):
    cap = cv2.VideoCapture(video_path)
    buffer = []
    predictions = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            lm = extract_landmarks_from_frame(results)
            buffer.append(lm)
            if len(buffer) >= SEQ_LEN:
                seq = np.array(buffer[-SEQ_LEN:])[None, ...]  # shape (1, SEQ_LEN, FEATURE_DIM)
                prob = model.predict(seq, verbose=0)[0]
                idx = int(np.argmax(prob))
                label = labels_map[idx]
                conf = float(np.max(prob))
                predictions.append((label, conf))
                # overlay on frame
                cv2.putText(frame, f"{label} {conf:.2f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow("Predict", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    return predictions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    model = load_model(MODEL_PATH)
    labels_map = load_label_map(LABELS_PATH)
    inv_labels = {v: k for k, v in labels_map.items()}
    # labels_map loaded is idx->label if we saved that way; adjust depending on your saved structure
    preds = predict_on_video(args.video, model, labels_map)
    # Print simple summary: frequent predicted label
    if preds:
        agg = {}
        for p, c in preds:
            agg[p] = agg.get(p, 0) + 1
        sorted_agg = sorted(agg.items(), key=lambda x: -x[1])
        print("Most frequent predictions:", sorted_agg[:5])
    else:
        print("No predictions made.")
