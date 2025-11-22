# infer_image.py
# Run RetinaNet (KerasCV) fire/smoke detection on a single image.
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



def preprocess_image(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = img_rgb.astype("float32") / 255.0
    return img_bgr, np.expand_dims(img_rgb, axis=0)


def run_inference(model, image_path):
    original_img, input_tensor = preprocess_image(image_path)

    # ⛔ DO NOT pass {"images": input_tensor} to predict()
    # ✅ For inference, KerasCV RetinaNet expects just the image tensor.
    preds = model.predict(input_tensor, verbose=0)

    # preds is a dict: {"boxes", "classes", "confidence"}
    boxes = preds["boxes"][0]        # (N, 4) in rel_xyxy
    classes = preds["classes"][0]    # (N,)
    scores = preds["confidence"][0]  # (N,)

    h, w, _ = original_img.shape

    fire_detected = False
    smoke_detected = False

    for box, cls_id, score in zip(boxes, classes, scores):
        if score < CONF_THRESH:
            continue

        if int(cls_id) == 0:
            fire_detected = True
        elif int(cls_id) == 1:
            smoke_detected = True

        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        label_name = CLASSES[int(cls_id)]
        label = f"{label_name} {score:.2f}"

        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            original_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Final printed message
    if fire_detected and smoke_detected:
        print("🔥 Fire and 💨 Smoke detected!")
    elif fire_detected:
        print("🔥 Fire detected!")
    elif smoke_detected:
        print("💨 Smoke detected!")
    else:
        print("✔ No fire or smoke detected.")

        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        cls_id = int(cls_id)
        label_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
        label = f"{label_name} {score:.2f}"

        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            original_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Save result to file (no imshow, since your OpenCV has no GUI backend)
    base = os.path.basename(image_path)
    out_name = f"output_{base}"
    out_path = os.path.join(os.path.dirname(image_path), out_name)

    cv2.imwrite(out_path, original_img)
    print(f"Saved detection result to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer_image.py /path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model = load_model()
    run_inference(model, image_path)
