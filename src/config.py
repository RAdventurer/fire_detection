# config.py
# Configuration for Fire/Smoke RetinaNet Project

import os

# -------------------------------
# ROOT PATHS
# -------------------------------

# fire_detection/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fire_detection/data/
DATA_DIR = os.path.join(PROJECT_ROOT, "/home/ico/project/fire_detection/data")

# -------------------------------
# DATASET PATHS (your exact structure)
# -------------------------------

# Images
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")
VAL_IMAGES_DIR   = os.path.join(DATA_DIR, "val", "images")
TEST_IMAGES_DIR  = os.path.join(DATA_DIR, "test", "images")

# Labels
TRAIN_LABELS_DIR = os.path.join(DATA_DIR, "train", "labels")
VAL_LABELS_DIR   = os.path.join(DATA_DIR, "val", "labels")
TEST_LABELS_DIR  = os.path.join(DATA_DIR, "test", "labels")

# -------------------------------
# MODEL PATHS
# -------------------------------

# Directory where model + checkpoints will be stored
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "fire_retinanet")

# Final model saved as .keras file
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")

# -------------------------------
# MODEL SETTINGS
# -------------------------------

# Your dataset classes
CLASSES = ["fire", "smoke"]   # class 0 = fire, class 1 = smoke
NUM_CLASSES = len(CLASSES)

# Input image size (640x640)
IMAGE_SIZE = 640

# Detection threshold during inference
CONF_THRESH = 0.4

# RetinaNet bounding box format
BBOX_FORMAT = "rel_xyxy"
