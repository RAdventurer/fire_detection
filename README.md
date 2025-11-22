# Fire & Smoke Detection (TensorFlow)
This project detects fire and smoke in images and videos using a TensorFlow model.

# Fire & Smoke Detection (TensorFlow)

This project detects **fire** and **smoke** in images and videos using a TensorFlow model.

## Structure

- `data/`
  - `images/train`, `images/val` – training and validation images
  - `labels/train`, `labels/val` – YOLO-format label files
- `src/`
  - `config.py` – paths and constants
  - `dataset_prep.py` – YOLO → TFRecord converter
  - `train.py` – simple training script (one box per image)
  - `infer_image.py` – run detection on one image
  - `infer_video.py` – run detection on video/webcam
- `models/fire_saved_model/` – exported TensorFlow SavedModel

## Basic Usage

1. Install dependencies:

```bash
pip install -r requirements.txt


# 1) install deps
pip install -r requirements.txt

# 2) train RetinaNet
python src/train.py

# 3) test on image
python src/infer_image.py path/to/test.jpg

# 4) test on webcam
python src/infer_video.py
