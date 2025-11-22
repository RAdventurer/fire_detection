from huggingface_hub import hf_hub_download
from src.config import MODEL_PATH, MODEL_DIR
import os

REPO_ID = "Radventure/fire_detection"
FILENAME = "final_model.keras"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("⬇️ Downloading model from HuggingFace…")

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="model"
    )

    os.system(f"cp {downloaded} {MODEL_PATH}")
    print("✅ Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    download_model()
