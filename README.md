# CamAI - Production-Grade Office Surveillance Person Detection

Production-grade Python system for counting people from 2 RTSP office cameras with **different scene conditions**. Uses YOLOv8m with per-camera detection settings.

## Scene Conditions

- **cam1**: Medium density, partially occluded people (desks, monitors)
- **cam2**: High density, small people, top-down view

## Per-Camera Settings

| Camera | imgsz | conf | iou | Use case |
|--------|-------|------|-----|----------|
| cam1   | 1600  | 0.12 | 0.6 | Occlusions |
| cam2   | 1920  | 0.06 | 0.6 | Small people |

## Features

- **YOLOv8m** - Better for small + occluded people (vs nano)
- **Per-camera config** - Different imgsz/conf per scene
- **Debug output** - Prints detected classes and confidence scores
- **RTSP**: cv2.CAP_FFMPEG, buffer=1, 3 retries, timeout
- **Single frame** - No streaming, capture only when needed

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app/main.py
```

After start:
- Scheduler runs in background (capture/detect interval: see `INTERVAL_SECONDS` in `config.py`, default 300 s)
- Web dashboard is available at [http://localhost:5000](http://localhost:5000)
- Swagger docs: [http://localhost:5000/swagger](http://localhost:5000/swagger)

## Output Structure

```
outputs/
├── raw/           # Original screenshots
├── processed/     # Boxes + "Total: X people"
├── crops/         # One crop per detected person
└── logs/
    └── stats.json # JSON records (appended)
```

## Configuration

Edit `config.py`:
- `RTSP_CAMERAS` - Camera URLs
- `CAMERA_DETECTION_SETTINGS` - Per-camera imgsz, conf
- `INTERVAL_SECONDS` - Capture interval in seconds (default: 300)

## Requirements

- Python 3.8+
- ultralytics, opencv-python, flask, flasgger, requests
- Network access to RTSP cameras
