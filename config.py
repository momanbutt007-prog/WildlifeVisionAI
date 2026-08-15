import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "wildlife_classifier_final.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

IMAGE_SIZE = 384
MAX_CONTENT_LENGTH = 10 * 1024 * 1024
DEFAULT_CLASSES = ["buffalo", "elephant", "rhino", "zebra"]
SECRET_KEY = os.environ.get("SECRET_KEY", "wildlife-vision-ai-secret")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
