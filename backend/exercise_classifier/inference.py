import tensorflow as tf
import numpy as np
import cv2
import json

# Load the pre-trained model and labels
MODEL_PATH = "models/best_model.h5"
LABELS_PATH = "models/labels_map.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels_map = json.load(f)

def classify_exercise(image_path):
    # Preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    return labels_map[str(class_idx)]