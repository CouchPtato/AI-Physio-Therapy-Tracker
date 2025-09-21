# train_model.py
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from collections import defaultdict

SEQ_DIR = "data/sequences"
MODEL_DIR = "models"
SEQ_LEN = 60
FEATURE_DIM = 33 * 4  # must match collect_data FEATURE_DIM

def load_data(seq_dir):
    files = glob.glob(os.path.join(seq_dir, "*.npy"))
    X = []
    y = []
    labels_map = {}
    label_idx = 0
    for f in files:
        base = os.path.basename(f)
        # label assumed prefix before first underscore
        label = base.split("_")[0]
        if label not in labels_map:
            labels_map[label] = label_idx
            label_idx += 1
        X.append(np.load(f))
        y.append(labels_map[label])
    X = np.array(X, dtype=np.float32)  # (N, SEQ_LEN, FEATURE_DIM)
    y = np.array(y, dtype=np.int32)
    return X, y, labels_map

def build_model(seq_len, feat_dim, num_classes):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(seq_len, feat_dim)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, labels_map = load_data(SEQ_DIR)
    print("Loaded", X.shape, y.shape, "labels:", labels_map)
    if len(X) == 0:
        raise SystemExit("No training data. Run data collection first.")
    y_cat = to_categorical(y, num_classes=len(labels_map))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    model = build_model(SEQ_LEN, FEATURE_DIM, num_classes=len(labels_map))
    ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), monitor="val_accuracy", save_best_only=True, verbose=1)
    early = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=80, batch_size=16, callbacks=[ckpt, early])
    # save label map
    import json
    with open(os.path.join(MODEL_DIR, "labels_map.json"), "w") as f:
        json.dump(labels_map, f)
    print("Training finished. Model saved.")
