# src/main.py
# FastAPI + HTML dashboard for RetinaNet Fire/Smoke detection.

import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from tensorflow import keras

from src.config import MODEL_PATH, IMAGE_SIZE, CONF_THRESH, CLASSES
from src.download_model import download_model

app = FastAPI(title="Fire & Smoke Detection API")

# -------------------------
# Ensure model exists, then load
# -------------------------

if not os.path.exists(MODEL_PATH):
    print("Model not found, downloading from Hugging Face...")
    download_model()

print(f"Loading RetinaNet model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")


# -------------------------
# Helper functions
# -------------------------

def preprocess_bytes(image_bytes: bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = img_rgb.astype("float32") / 255.0
    input_tensor = np.expand_dims(img_rgb, axis=0)
    return img_bgr, input_tensor


def run_model_on_bytes(image_bytes: bytes):
    original_img, input_tensor = preprocess_bytes(image_bytes)
    preds = model.predict(input_tensor, verbose=0)

    boxes = preds["boxes"][0]
    classes = preds["classes"][0]
    scores = preds["confidence"][0]

    h, w, _ = original_img.shape

    detections = []
    fire_detected = False
    smoke_detected = False

    for box, cid, score in zip(boxes, classes, scores):
        score = float(score)
        if score < CONF_THRESH:
            continue

        cid = int(cid)
        if cid == 0:
            fire_detected = True
        elif cid == 1:
            smoke_detected = True

        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        cls_name = CLASSES[cid] if 0 <= cid < len(CLASSES) else str(cid)

        detections.append({
            "class_id": cid,
            "class_name": cls_name,
            "score": round(score, 3),
            "box_rel": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "box_xyxy": [x1, y1, x2, y2],
        })

    return {
        "fire_detected": fire_detected,
        "smoke_detected": smoke_detected,
        "detections": detections,
    }, original_img, boxes, classes, scores


# -------------------------
# HTML Dashboard
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <head>
        <title>Fire & Smoke Detection</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 30px auto; }
          h1 { color: #333; }
          form { margin-bottom: 20px; }
          button { margin-left: 10px; padding: 4px 10px; }
          .status { margin-top: 20px; padding: 10px; border-radius: 6px; }
          .ok { background: #e7f7e7; }
          .fire { background: #ffe6e6; }
          .smoke { background: #e6f0ff; }
          .both { background: #fff3cd; }
          .detections { margin-top: 10px; font-size: 13px; white-space: pre-wrap; }
          img { margin-top: 15px; max-width: 100%; border: 1px solid #ccc; }
        </style>
      </head>
      <body>
        <h1>🔥 Fire & 💨 Smoke Detection</h1>
        <p>Upload an image. The model will say if there is <b>fire</b>, <b>smoke</b>, or <b>nothing</b>, and show detections.</p>

        <form id="upload-form">
          <input type="file" name="file" id="file-input" accept="image/*" required />
          <button type="submit">Detect</button>
        </form>

        <div id="result" class="status" style="display:none;"></div>
        <pre id="raw-json" class="detections"></pre>
        <h3>Output image:</h3>
        <img id="output-img" src="" style="display:none;" />

        <script>
          const form = document.getElementById('upload-form');
          const resultDiv = document.getElementById('result');
          const rawJson = document.getElementById('raw-json');
          const outputImg = document.getElementById('output-img');

          form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('file-input');
            if (!fileInput.files.length) {
              alert('Please choose an image first.');
              return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            resultDiv.style.display = 'block';
            resultDiv.className = 'status';
            resultDiv.textContent = 'Running detection...';
            rawJson.textContent = '';
            outputImg.style.display = 'none';
            outputImg.src = '';

            try {
              // 1) JSON
              const resp = await fetch('/predict', {
                method: 'POST',
                body: formData
              });
              const data = await resp.json();
              rawJson.textContent = JSON.stringify(data, null, 2);

              let msg = '';
              let css = 'status ';
              if (data.fire_detected && data.smoke_detected) {
                  msg = '🔥 Fire AND 💨 Smoke detected!';
                  css += 'both';
              } else if (data.fire_detected) {
                  msg = '🔥 Fire detected!';
                  css += 'fire';
              } else if (data.smoke_detected) {
                  msg = '💨 Smoke detected!';
                  css += 'smoke';
              } else {
                  msg = '✔ No fire or smoke detected.';
                  css += 'ok';
              }
              resultDiv.className = css;
              resultDiv.textContent = msg;

              // 2) Image
              const imgResp = await fetch('/predict_image', {
                  method: 'POST',
                  body: formData
              });
              if (imgResp.ok) {
                  const imgBlob = await imgResp.blob();
                  const imgURL = URL.createObjectURL(imgBlob);
                  outputImg.style.display = 'block';
                  outputImg.src = imgURL;
              }

            } catch (err) {
              console.error(err);
              resultDiv.className = 'status';
              resultDiv.textContent = 'Error during detection.';
            }
          });
        </script>
      </body>
    </html>
    """


# -------------------------
# JSON prediction endpoint
# -------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result, _, _, _, _ = run_model_on_bytes(contents)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------
# Image prediction endpoint
# -------------------------

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result, original_img, boxes, classes, scores = run_model_on_bytes(contents)

        h, w, _ = original_img.shape
        vis_img = original_img.copy()

        for box, cid, score in zip(boxes, classes, scores):
            score = float(score)
            if score < CONF_THRESH:
                continue

            cid = int(cid)
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            cls_name = CLASSES[cid] if 0 <= cid < len(CLASSES) else str(cid)
            label = f"{cls_name} {score:.2f}"

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        ok, buffer = cv2.imencode(".jpg", vis_img)
        if not ok:
            raise RuntimeError("Failed to encode image")

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
