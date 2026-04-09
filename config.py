import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

API_KEY = os.environ.get("API_KEY", "abc123")

# API keys for different services
API_KEYS_FILE = OUTPUTS_DIR / "logs" / "api_keys.json"

# RTSP cameras are not used for API-only mode.
# Keep this empty when you do not need real-time camera capture.
RTSP_CAMERAS = {}

INTERVAL_SECONDS = 300

RTSP_TIMEOUT_SECONDS = 15
RTSP_RETRY_ATTEMPTS = 3

YOLO_MODEL = "yolov8n.pt"
PERSON_CLASS_ID = 0
IOU_THRESHOLD = 0.45
YOLO_AUGMENT = False
YOLO_MAX_DET = 100
MIN_BOX_AREA_RATIO = 0.001
MIN_BOX_HEIGHT_RATIO = 0.05

CAMERA_DETECTION_SETTINGS = {
    # For arbitrary internet images: balanced thresholds (good recall, fewer false positives).
    "internet": {
        "imgsz": 640,
        "conf": 0.3,
        "dedupe_iou": 0.65,
        "min_aspect_ratio": 0.85,
    },
    "cam1": {
        "imgsz": 640,
        "conf": 0.3,
        "dedupe_iou": 0.65,
        "min_aspect_ratio": 0.85,
    },
    "cam2": {
        "imgsz": 640,
        "conf": 0.3,
        "dedupe_iou": 0.65,
        "min_aspect_ratio": 0.85,
    },
    # API v1 tasks often send `camera_name` as camera_id (e.g. "Bosh xona").
    "Bosh xona": {
        "imgsz": 640,
        "conf": 0.35,
        "dedupe_iou": 0.65,
        "min_aspect_ratio": 1.15,
    },
}

RAW_IMAGES_DIR = OUTPUTS_DIR / "raw"
PROCESSED_IMAGES_DIR = OUTPUTS_DIR / "processed"
CROPS_DIR = OUTPUTS_DIR / "crops"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
UPLOADS_PROCESSED_DIR = OUTPUTS_DIR / "uploads_processed"
LOGS_DIR = OUTPUTS_DIR / "logs"
STATS_FILE = LOGS_DIR / "stats.json"
