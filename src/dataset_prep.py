# dataset_prep.py
# Create tf.data.Dataset for KerasCV RetinaNet from YOLO-format labels.

import os
import glob
import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    TEST_LABELS_DIR,
    IMAGE_SIZE,
    BBOX_FORMAT,
)


AUTOTUNE = tf.data.AUTOTUNE
MAX_BOXES = 50  # max objects per image, pad to this size


def yolo_to_rel_xyxy(xc, yc, w, h):
    """
    YOLO -> rel_xyxy conversion.
    xc, yc, w, h are already relative [0,1].
    """
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return x1, y1, x2, y2


def load_image_and_labels(img_path, label_path):
    """Load an image and its YOLO txt labels, return padded boxes/classes."""
    # --- image ---
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0

    # --- boxes ---
    boxes = []
    classes = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cid, xc, yc, w, h = map(float, parts)
                cid = int(cid) - 1   # convert 1→0, 2→1
                x1, y1, x2, y2 = yolo_to_rel_xyxy(xc, yc, w, h)
                boxes.append([x1, y1, x2, y2])
                classes.append(cid)


    # convert to numpy
    if len(boxes) == 0:
        boxes = np.zeros((0, 4), dtype="float32")
        classes = np.zeros((0,), dtype="int32")
    else:
        boxes = np.array(boxes, dtype="float32")
        classes = np.array(classes, dtype="int32")

    # --- PAD to fixed size MAX_BOXES ---
    padded_boxes = np.zeros((MAX_BOXES, 4), dtype="float32")
    padded_classes = -np.ones((MAX_BOXES,), dtype="int32")  # -1 = no object

    n = min(len(boxes), MAX_BOXES)
    if n > 0:
        padded_boxes[:n] = boxes[:n]
        padded_classes[:n] = classes[:n]

    return img, padded_boxes, padded_classes


def make_file_list(images_dir, labels_dir):
    img_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    label_paths = []
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label_paths.append(os.path.join(labels_dir, base + ".txt"))
    return img_paths, label_paths


def numpy_loader(img_path, label_path):
    img, boxes, classes = load_image_and_labels(img_path, label_path)
    return img, boxes, classes


def tf_loader(img_path, label_path):
    img, boxes, classes = tf.numpy_function(
        numpy_loader,
        inp=[img_path, label_path],
        Tout=[tf.float32, tf.float32, tf.int32],
    )

    img.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    boxes.set_shape((MAX_BOXES, 4))
    classes.set_shape((MAX_BOXES,))

    bounding_boxes = {
        "boxes": boxes,
        "classes": classes,
        "bounding_box_format": tf.constant(BBOX_FORMAT),
    }

    # important: return dict with "images" AND "bounding_boxes"
    example = {
        "images": img,
        "bounding_boxes": bounding_boxes,
    }
    return example


def make_dataset(images_dir, labels_dir, batch_size=4, shuffle=True):
    img_paths, label_paths = make_file_list(images_dir, labels_dir)

    ds_img = tf.data.Dataset.from_tensor_slices(img_paths)
    ds_lbl = tf.data.Dataset.from_tensor_slices(label_paths)
    ds = tf.data.Dataset.zip((ds_img, ds_lbl))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths))

    ds = ds.map(tf_loader, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def get_train_val_datasets(batch_size=4):
    train_ds = make_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, batch_size=batch_size, shuffle=True)
    val_ds   = make_dataset(VAL_IMAGES_DIR,   VAL_LABELS_DIR,   batch_size=batch_size, shuffle=False)
    return train_ds, val_ds


def get_test_dataset(batch_size=4):
    test_ds = make_dataset(TEST_IMAGES_DIR, TEST_LABELS_DIR, batch_size=batch_size, shuffle=False)
    return test_ds


if __name__ == "__main__":
    train_ds, val_ds = get_train_val_datasets(batch_size=2)
    for batch in train_ds.take(1):
        print("Keys:", batch.keys())
        print("Images:", batch["images"].shape)
        print("Boxes:", batch["bounding_boxes"]["boxes"].shape)
        print("Classes:", batch["bounding_boxes"]["classes"].shape)
