# CamAI - Production-Grade Office Surveillance Person Detection

Production-grade Python system for counting people from 2 RTSP office cameras with **different scene conditions**. Uses YOLOv8n with optimized settings for fast API processing.

## Performance Optimizations

- **YOLOv8n** - Fast nano model (vs YOLOv8m)
- **640x640 imgsz** - Optimized resolution for speed
- **Image resizing** - Large images (>1920px) auto-resized
- **Confidence threshold** - 0.3 for faster filtering
- **Max detections** - Limited to 100 for speed
- **API response time** - ~3-4 seconds per image

## Scene Conditions

- **cam1**: Medium density, partially occluded people (desks, monitors)
- **cam2**: High density, small people, top-down view

## Per-Camera Settings

| Camera | imgsz | conf | iou | Use case |
|--------|-------|------|-----|----------|
| cam1   | 640   | 0.3  | 0.45| Fast API |
| cam2   | 640   | 0.3  | 0.45| Fast API |

## Features

- **YOLOv8n** - Optimized for speed (vs YOLOv8m)
- **Per-camera config** - Different imgsz/conf per scene
- **Debug output** - Prints detected classes and confidence scores
- **RTSP**: cv2.CAP_FFMPEG, buffer=1, 3 retries, timeout
- **Single frame** - No streaming, capture only when needed
- **API-only mode** - Disabled real-time RTSP for API usage
- **Service-based auth** - Different API keys per service

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app/main.py
```

After start:
- Web dashboard is available at [http://localhost:5000](http://localhost:5000)
- API test form: [http://localhost:5000/api/test-submit](http://localhost:5000/api/test-submit)
- Swagger docs: [http://localhost:5000/swagger](http://localhost:5000/swagger)

## API Usage

### Submit Image for Detection

```bash
curl -X POST http://localhost:5000/api/v1/tasks/submit \
  -H "X-API-Key: abc123" \
  -F "file=@image.jpg" \
  -F 'metadata={"service":"default","camera_name":"test"}'
```

Response:
```json
{
  "service": "default",
  "camera_name": "test",
  "count": 7
}
```

### Service-Based Authentication

Each service has its own API key stored in `outputs/logs/api_keys.json`:

```json
{
  "default": "abc123",
  "service1": "key456",
  "service2": "key789"
}
```

## Deploy on server

1. `cd` loyiha papkasiga.
2. Python virtual muhit yarating va faollashtiring:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Kerakli paketlarni o‘rnating:

```bash
pip install -r requirements.txt
```

4. `config.py` faylini o‘zgartiring, RTSP URL va kamera sozlamalarini to‘g‘rilang.
5. (`API_KEY` bo‘lsa) agar kerak bo‘lsa `API_KEY` o‘rnatish mumkin:

```bash
export API_KEY=abc123
```

6. Serverda dastur ishga tushiring:

```bash
python app/main.py
```

Agar fon rejimida ishga tushirish kerak bo‘lsa:

```bash
nohup python app/main.py > outputs/logs/server.log 2>&1 &
```

### Systemd xizmat sifatida ishga tushirish

1. `camai.service` faylini serverda `/etc/systemd/system/` ga nusxa ko‘chiring.
2. Ichidagi `WorkingDirectory` va `ExecStart` yo‘lini o‘z loyihangiz joyiga moslab o‘zgartiring.
3. Xizmatni yuklang va ishga tushiring:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now camai.service
sudo systemctl status camai.service
```

4. Loglar uchun:

```bash
sudo journalctl -u camai.service -f
```

5. Brauzerda `http://<server-ip>:5000` manziliga kirib tekshiring.

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
