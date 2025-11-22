# infer_video.py
# Run RetinaNet (KerasCV) fire/smoke detection on video or webcam.

import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_cv

from config import MODEL_DIR, IMAGE_SIZE, CONF_THRESH, CLASSES, BBOX_FORMAT


def load_model():
    print(f"Loading RetinaNet from {MODEL_DIR}...")
    model = keras.models.load_model(MODEL_DIR)
    return model


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = rgb.astype("float32") / 255.0
    return np.expand_dims(rgb, axis=0)


def draw_detections(frame, boxes, classes, scores):
    h, w, _ = frame.shape

    for box, cls_id, score in zip(boxes, classes, scores):
        if score < CONF_THRESH:
            continue

        # rel_xyxy → pixel
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        cls_id = int(cls_id)
        label_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
        label = f"{label_name} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return frame


def detect_video(source, model):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: cannot open video source:", source)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess_frame(frame)
        preds = model.predict(inp, verbose=0)
        boxes = preds["boxes"][0]
        classes = preds["classes"][0]
        scores = preds["confidence"][0]

        frame_out = draw_detections(frame, boxes, classes, scores)
        cv2.imshow("Fire & Smoke Detection - Video", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Usage:
    #   python infer_video.py           # webcam
    #   python infer_video.py video.mp4 # video file

    model = load_model()

    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = 0  # webcam

    detect_video(src, model)
