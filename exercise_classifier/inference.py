import cv2
import numpy as np
import mediapipe as mp
import argparse
import tensorflow as tf

# ================== CONFIG ==================
sequence_length = 60  # must match training sequence length
class_names = ["squat"]  # change to match your training labels

# ================== HELPER FUNCTIONS ==================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    """Extracts pose landmarks into a flat numpy array"""
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks)
    else:
        return np.zeros(33 * 4)  # 33 landmarks × (x,y,z,visibility)

# ================== INFERENCE ==================
def run_inference(video_path, model_path):
    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Open video (or webcam if '0')
    cap = cv2.VideoCapture(0 if video_path == "0" else video_path)

    sequence = []  # store recent frames for sequence input

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Extract landmarks and add to sequence
            landmarks = extract_landmarks(results)
            sequence.append(landmarks)
            if len(sequence) > sequence_length:
                sequence = sequence[-sequence_length:]  # keep last 60 frames

            # Predict only if we have full sequence
            predicted_class = None
            confidence = None
            if len(sequence) == sequence_length:
                input_data = np.expand_dims(sequence, axis=0)  # shape (1, 60, 132)
                prediction = model.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

            # Draw landmarks
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display predicted class
            if predicted_class is not None:
                text = f"{class_names[predicted_class]} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Exercise Recognition", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# ================== MAIN ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file or '0' for webcam")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .h5 model")
    args = parser.parse_args()

    run_inference(args.video, args.model)
