# Intelligent Surveillance for Telecom Sites  
## Video Module – Fire & Smoke Detection (Technical Test)

This repository implements the **video anomaly detection module** for the technical test:

> **“Système Intelligent de Surveillance Multi-Modal pour Infrastructures Télécom” – Digitup Company**

The full requested system includes:
- Video analysis
- IoT sensor analysis
- Network KPI analysis
- Multi-modal fusion + alerting

👉 **This repo focuses only on the video part**:  
real-time detection of **fire** and **smoke** on telecom/industrial sites, with a deployable **FastAPI + Docker** service.

---

## 1. Project Overview

### Goals

- Detect **fire** and **smoke** in CCTV-like images  
- Provide a **REST API** + simple **web UI**  
- Containerize the solution with **Docker**  
- Set up basic **CI/CD** with **GitHub Actions**  
- Store the model on **Hugging Face Hub** (no large files in Git)

### Tech Stack

- **Python 3.12**
- **TensorFlow + KerasCV (RetinaNet)**
- **FastAPI** + **Uvicorn**
- **OpenCV**
- **Docker**
- **GitHub Actions**
- **Hugging Face Hub** (model hosting)

---

## 2. Dataset

For this test, I use a **public Fire & Smoke dataset**, which is close to the telecom use case:

- ~17k images
- Bounding boxes in YOLO format
- Two classes:
  - `0 = fire`
  - `1 = smoke`
- Images from CCTV, industrial, indoor/outdoor scenes

The dataset is **not included** in this repo (ignored by `.gitignore`).

Expected structure:

```text
data/
  train/
    images/
    labels/
  val/
  
    images/
    labels/
  test/
    images/
    labels/

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
