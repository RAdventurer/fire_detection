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

# Fire & Smoke Detection – RetinaNet + KerasCV

This repository contains a **fire & smoke detection pipeline** based on:

- **TensorFlow / Keras**
- **KerasCV RetinaNet detector**
- **YOLO-style fire/smoke dataset**
- **FastAPI** web API + simple dashboard
- **Docker** + **GitHub Actions** + **Hugging Face Hub** for the model

The goal of this project is to demonstrate an **end-to-end process**:
from dataset → training → saving → serving via API → containerization and CI/CD.

It is designed as a first step for a **multi-modal telecom site monitoring system**.  
For now, this repository focuses only on the **video / image modality** (fire & smoke).

> ⚠️ **Important:**  
> The model in this repository was trained for **only 10 epochs**, mainly to test the full MLOps process.  
> It is **not optimized for maximum accuracy** and is **not production-ready**.  
> Use it as a **technical demo** and a starting point, not as a final safety system.

---

## 1. Context and Use Case

The long-term objective is to build an **intelligent surveillance system** for telecom sites:

- Detect **physical risks** like fire or smoke near antennas and equipment
- Later: fuse this with **sensor** and **network KPI** data

This repo covers the **fire & smoke detection** part:

- Input: images (or frames extracted from CCTV / surveillance cameras)
- Output: bounding boxes for fire and smoke, plus a simple decision:
  - `FIRE detected`
  - `SMOKE detected`
  - `FIRE + SMOKE detected`
  - `Nothing detected`

---

## 2. Project Structure

```text
fire_detection/
  data/                      # local dataset (ignored in git)
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/

  models/
    fire_retinanet/
      final_model.keras      # downloaded from Hugging Face

  src/
    __init__.py
    config.py                # paths, image size, thresholds, classes
    dataset_prep.py          # TF dataset builder from YOLO labels
    train.py                 # train RetinaNet
    infer_image.py           # CLI: single image inference
    infer_video.py           # CLI: video / webcam inference
    download_model.py        # download model from Hugging Face Hub
    main.py                  # FastAPI app + web dashboard

  Dockerfile                 # container build
  requirements.txt           # Python dependencies
  .github/
    workflows/
      docker-ci.yml          # CI/CD: build & push Docker image
  README.md


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

```
## 5. Model Storage and Hugging Face Hub
```
The trained model is stored on Hugging Face:

Repo ID: Radventure/fire_detection

File: final_model.keras

File: src/download_model.py uses huggingface-hub:

Downloads final_model.keras from Radventure/fire_detection

Saves it to: models/fire_retinanet/final_model.keras

src/main.py:

Checks if MODEL_PATH exists (models/fire_retinanet/final_model.keras)

If not, calls download_model()

Loads the model with keras.models.load_model(...)

So you do not need to manually copy the model file.
```
## 6. Running Locally (Python, no Docker)
```
6.1. Create and activate virtual environment
cd ~/project/fire_detection

python3 -m venv .venv
source .venv/bin/activate

6.2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


Make sure requirements.txt includes at least:

fastapi
uvicorn
tensorflow
keras
keras-cv
opencv-python
opencv-python-headless
huggingface-hub
numpy


(and any others you need)

6.3. Run the FastAPI app
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000


On first run:

The app will download the model from Hugging Face (if missing)

Then it will start the API and basic dashboard

Open in your browser:

http://localhost:8000


You will see:

A simple web UI

A file upload field

A status box:

“🔥 Fire detected”

“💨 Smoke detected”

“🔥 Fire AND 💨 Smoke detected”

“✔ No fire or smoke detected”

Raw JSON detections

Output image with bounding boxes
```
## 7. Running with Docker (locally)
```
7.1. Install Docker (Ubuntu quick version)
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y


Give your user Docker permissions:

sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker


Test:

docker ps

7.2. Pull image from GitHub Container Registry

The GitHub Action builds and pushes this image:

ghcr.io/radventurer/fire-detector:latest


Pull it:

docker pull ghcr.io/radventurer/fire-detector:latest


For a public image, you do not need to log in.

7.3. Run the container
docker run -p 8000:8000 ghcr.io/radventurer/fire-detector:latest


Then open:

http://localhost:8000


The image already includes:

All Python dependencies

The model downloaded from Hugging Face (via download_model.py during build)

The FastAPI app

To run in background:

docker run -d -p 8000:8000 ghcr.io/radventurer/fire-detector:latest
```
## 8. Training Script (optional)
```
If you want to retrain:

Prepare your dataset under data/train, data/val, data/test

Check or adjust src/config.py for:

IMAGE_SIZE

CLASSES

dataset paths

Run:

source .venv/bin/activate  # if using venv
python src/train.py


This will:

Build TF datasets from YOLO labels (dataset_prep.py)

Train RetinaNet for the configured number of epochs

Save the model to: models/fire_retinanet/final_model.keras

You can then upload the new model to Hugging Face if you want a better version.

Note: For serious use, you should:

Train for more epochs

Do proper validation and hyperparameter tuning

Evaluate on a dedicated test set and measure recall/precision

Possibly use a larger / better dataset
```
## 9. API Summary
```
GET /

Returns the HTML dashboard

Upload image → see status + output image + JSON

POST /predict

Input: multipart/form-data with field file

Output: JSON example:

{
  "fire_detected": true,
  "smoke_detected": false,
  "detections": [
    {
      "class_id": 0,
      "class_name": "fire",
      "score": 0.93,
      "box_rel": [0.12, 0.30, 0.45, 0.60],
      "box_xyxy": [80, 100, 240, 320]
    }
  ]
}

POST /predict_image

Input: multipart/form-data with field file

Output: JPEG image with bounding boxes drawn
```
## 10. CI/CD with GitHub Actions
```
File: .github/workflows/docker-ci.yml

On push to main, it:

Checks out the repo

Builds Docker image from Dockerfile

Runs python -m src.download_model inside the image

Pushes the image to:

ghcr.io/radventurer/fire-detector:latest


You can then deploy this container image on:

Any virtual machine (Docker installed)

Render, Railway, Kubernetes, or cloud services
```
## 11. Limitations and Next Steps
```
The model was trained for only 10 epochs, so:

Accuracy and recall are limited

It is suitable for technical demo, not for production safety

No calibration, no drift monitoring yet

No multi-modal fusion (only vision)

Possible improvements:

Train longer with better hyperparameters

Add more robust augmentation

Evaluate properly on a held-out test set

Integrate with:

Time-series anomaly detection (sensors)

Network KPI analysis (telecom metrics)

Add real-time video ingestion and alerting logic
```
This repository is mainly a professional template to show:

- How to structure a detection project

- How to train and store a model

- How to serve it with FastAPI

- How to containerize and automate builds with GitHub Actions

- How to integrate a Hugging Face model into the pipeline