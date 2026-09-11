import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
MODEL_PATH = MODELS_DIR / "modelo_yoga_kaggle.h5"
LABELS_PATH = MODELS_DIR / "labels.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# Origenes permitidos para CORS (coma-separados). Por defecto, el frontend estático local.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000"
    ).split(",")
    if origin.strip()
]
