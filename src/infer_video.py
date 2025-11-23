# src/infer_video.py
# Run RetinaNet (KerasCV) fire/smoke detection on video file.

import sys
import os
import cv2
import numpy as np
from tensorflow import keras

from src.config import MODEL_PATH, IMAGE_SIZE, CONF_THRESH, CLASSES, BBOX_FORMAT


def load_model():
    print(f"Loading RetinaNet from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
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


def detect_video_file(source_path, model, output_path):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print("Error: cannot open video source:", source_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = 0
    fire_frames = 0
    smoke_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        inp = preprocess_frame(frame)
        preds = model.predict(inp, verbose=0)

        boxes = preds["boxes"][0]
        classes = preds["classes"][0]
        scores = preds["confidence"][0]

        # count detections for statistics
        fire_here = False
        smoke_here = False
        for box, cid, score in zip(boxes, classes, scores):
            if score < CONF_THRESH:
                continue
            cid = int(cid)
            if cid == 0:
                fire_here = True
            elif cid == 1:
                smoke_here = True

        if fire_here:
            fire_frames += 1
        if smoke_here:
            smoke_frames += 1

        frame_out = draw_detections(frame, boxes, classes, scores)
        writer.write(frame_out)

    cap.release()
    writer.release()

    print(f"Processed video saved to: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with fire: {fire_frames}")
    print(f"Frames with smoke: {smoke_frames}")


if __name__ == "__main__":
    # Usage:
    #   python -m src.infer_video path/to/video.mp4

    if len(sys.argv) < 2:
        print("Usage: python -m src.infer_video path/to/video.mp4")
        sys.exit(1)

    src_path = sys.argv[1]
    if not os.path.exists(src_path):
        print(f"Video not found: {src_path}")
        sys.exit(1)

    base = os.path.basename(src_path)
    out_name = f"output_{base}"
    out_path = os.path.join(os.path.dirname(src_path), out_name)

    model = load_model()
    detect_video_file(src_path, model, out_path)
