from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set")

api = HfApi(token=HF_TOKEN)

# ⚠️ Use your HF username here (Radventure, not RAdventurer)
REPO_ID = "Radventure/fire_detection"
MODEL_PATH = "/home/ico/project/fire_detection/models/fire_retinanet/final_model.keras"

# 1) Create the repo if it does not exist
api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,  # do nothing if already exists
)

# 2) Upload the model file
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="final_model.keras",
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"✅ Model uploaded to HuggingFace repo: {REPO_ID}")
