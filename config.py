from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

RTSP_CAMERAS = {
    "cam1": "rtsp://admin:Qwerty135@192.168.3.149:554/Streaming/Channels/101",
    "cam2": "rtsp://admin:Qwerty135@192.168.3.149:554/Streaming/Channels/201",
}

INTERVAL_SECONDS = 30

RTSP_TIMEOUT_SECONDS = 15
RTSP_RETRY_ATTEMPTS = 3

YOLO_MODEL = "yolov8m.pt"
PERSON_CLASS_ID = 0
IOU_THRESHOLD = 0.45
YOLO_AUGMENT = False
YOLO_MAX_DET = 300
MIN_BOX_AREA_RATIO = 0.000030
MIN_BOX_HEIGHT_RATIO = 0.012

CAMERA_DETECTION_SETTINGS = {
    "cam1": {
        "imgsz": 1600,
        "conf": 0.25,
        "dedupe_iou": 0.65,
    },
    "cam2": {
        "imgsz": 1920,
        "conf": 0.16,
        "dedupe_iou": 0.65,
    },
}

RAW_IMAGES_DIR = OUTPUTS_DIR / "raw"
PROCESSED_IMAGES_DIR = OUTPUTS_DIR / "processed"
CROPS_DIR = OUTPUTS_DIR / "crops"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
UPLOADS_PROCESSED_DIR = OUTPUTS_DIR / "uploads_processed"
LOGS_DIR = OUTPUTS_DIR / "logs"
STATS_FILE = LOGS_DIR / "stats.json"
