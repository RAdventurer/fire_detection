
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from tensorflow import keras

from src.config import MODEL_PATH, IMAGE_SIZE, CONF_THRESH, CLASSES
from src.download_model import download_model

app = FastAPI(title="Fire & Smoke Detection API")

# -------------------------------------------------------------------
# Ensure model exists, then load
# -------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("Model not found, downloading from Hugging Face...")
    download_model()

print(f"Loading RetinaNet model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# HTML Dashboard
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <head>
        <title>Fire & Smoke Detection</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 30px auto;
            background: #f5f5f5;
            color: #333;
          }
          h1 { text-align: center; }
          .card {
            background: #fff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-bottom: 25px;
          }
          .status {
            margin-top: 15px;
            padding: 10px;
            border-radius: 6px;
            font-weight: 600;
          }
          .ok { background: #e7f7e7; border:1px solid #7ac27a; color:#316a31; }
          .fire { background:#ffe0e0; border:1px solid #ff4b4b; color:#8b0000; }
          .smoke { background:#e6f0ff; border:1px solid #5c7edc; color:#20325c; }
          .both { background:#fff3cd; border:1px solid #ffcc00; color:#8a6d00; }
          .detections {
            margin-top: 10px;
            font-size: 13px;
            white-space: pre-wrap;
            background: #282c34;
            color: #e5e5e5;
            padding: 10px;
            border-radius: 6px;
          }
          img {
            margin-top: 15px;
            max-width: 100%;
            border-radius: 6px;
            border:1px solid #ccc;
          }
        </style>
      </head>

      <body>
        <h1>🔥 Fire & 💨 Smoke Detection</h1>

        <!-- Image Upload -->
        <div class="card">
          <h2>Image Detection</h2>
          <form id="img-form">
            <input type="file" id="img-input" accept="image/*" required>
            <button type="submit">Detect Image</button>
          </form>

          <div id="img-status" class="status" style="display:none;"></div>

          <h4>Detections (JSON)</h4>
          <pre id="img-json" class="detections"></pre>

          <h4>Output Image</h4>
          <img id="img-output" style="display:none;">
        </div>

        <!-- Video Upload -->
        <div class="card">
          <h2>Video Detection (Summary Only)</h2>
          <form id="vid-form">
            <input type="file" id="vid-input" accept="video/*" required>
            <button type="submit">Analyze Video</button>
          </form>

          <div id="vid-status" class="status" style="display:none;"></div>

          <h4>Video Summary (JSON)</h4>
          <pre id="vid-json" class="detections"></pre>
        </div>

        <script>
          // ---------------------------
          // IMAGE HANDLER
          // ---------------------------
          const imgForm = document.getElementById('img-form');
          imgForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const file = document.getElementById('img-input').files[0];
            if (!file) return;

            const fd = new FormData();
            fd.append("file", file);

            const status = document.getElementById('img-status');
            const outImg = document.getElementById('img-output');
            const raw = document.getElementById('img-json');

            status.style.display = "block";
            status.textContent = "Running detection...";

            // JSON prediction
            const resp = await fetch('/predict', { method:'POST', body:fd });
            const data = await resp.json();
            raw.textContent = JSON.stringify(data, null, 2);

            let cls = "status ";
            if (data.fire_detected && data.smoke_detected) {
              cls += "both";
              status.textContent = "🔥 Fire AND 💨 Smoke detected";
            }
            else if (data.fire_detected) {
              cls += "fire";
              status.textContent = "🔥 Fire detected";
            }
            else if (data.smoke_detected) {
              cls += "smoke";
              status.textContent = "💨 Smoke detected";
            }
            else {
              cls += "ok";
              status.textContent = "✔ No fire or smoke";
            }
            status.className = cls;

            // Image output
            const imgResp = await fetch('/predict_image', { method:'POST', body:fd });
            const blob = await imgResp.blob();
            outImg.src = URL.createObjectURL(blob);
            outImg.style.display = "block";
          });

          // ---------------------------
          // VIDEO HANDLER
          // ---------------------------
          const vidForm = document.getElementById('vid-form');
          vidForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const file = document.getElementById('vid-input').files[0];
            if (!file) return;

            const fd = new FormData();
            fd.append("file", file);

            const status = document.getElementById('vid-status');
            const raw = document.getElementById('vid-json');

            status.style.display = "block";
            status.textContent = "Processing video (sampling frames)...";

            const resp = await fetch('/predict_video_summary', {
              method: 'POST',
              body: fd
            });

            const data = await resp.json();
            raw.textContent = JSON.stringify(data, null, 2);

            let cls = "status ";
            if (data.status === "fire_and_smoke") {
              cls += "both";
              status.textContent = "🔥 + 💨 Detected in video!";
            }
            else if (data.status === "fire") {
              cls += "fire";
              status.textContent = "🔥 Fire detected in video!";
            }
            else if (data.status === "smoke") {
              cls += "smoke";
              status.textContent = "💨 Smoke detected in video!";
            }
            else {
              cls += "ok";
              status.textContent = "✔ No fire or smoke detected in video.";
            }
            status.className = cls;
          });
        </script>

      </body>
    </html>
    """



# -------------------------------------------------------------------
# JSON prediction endpoint
# -------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result, _, _, _, _ = run_model_on_bytes(contents)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------------
# Image prediction endpoint
# -------------------------------------------------------------------
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
import tempfile
import cv2
import numpy as np
# ... keep the rest of your imports and existing code ...


@app.post("/predict_video_summary")
async def predict_video_summary(file: UploadFile = File(...)):
    """
    Upload a video and get a summary:
    - total frames
    - how many frames with fire / smoke
    - global status
    """
    try:
        # Save uploaded video to a temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            os.remove(tmp_path)
            return JSONResponse({"error": "Cannot open video"}, status_code=400)

        total_frames = 0
        fire_frames = 0
        smoke_frames = 0

        # sample every Nth frame to reduce compute
        FRAME_STEP = 5

        while True:
            ret = cap.grab()
            if not ret:
                break

            # process every FRAME_STEP frames
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_idx % FRAME_STEP != 0:
                continue

            ret, frame = cap.retrieve()
            if not ret:
                break

            total_frames += 1

            # same preprocessing as images
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
            rgb = rgb.astype("float32") / 255.0
            inp = np.expand_dims(rgb, axis=0)

            preds = model.predict(inp, verbose=0)
            boxes = preds["boxes"][0]
            classes = preds["classes"][0]
            scores = preds["confidence"][0]

            fire_here = False
            smoke_here = False

            for box, cid, score in zip(boxes, classes, scores):
                if float(score) < CONF_THRESH:
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

        cap.release()
        os.remove(tmp_path)

        any_fire = fire_frames > 0
        any_smoke = smoke_frames > 0

        if any_fire and any_smoke:
            status = "fire_and_smoke"
        elif any_fire:
            status = "fire"
        elif any_smoke:
            status = "smoke"
        else:
            status = "none"

        return {
            "status": status,
            "sampled_frames": total_frames,
            "frames_with_fire": fire_frames,
            "frames_with_smoke": smoke_frames,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
