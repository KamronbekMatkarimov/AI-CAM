from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from collections.abc import Mapping, Sequence
import json
import threading
import queue
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory, request, Response
from flasgger import Swagger
import requests

import config
from app.detector import detect_persons
from app.scheduler import get_scheduler_state
from app.utils import read_stats_records


PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["TRAP_HTTP_EXCEPTIONS"] = True
app.config["DEBUG"] = True


@app.errorhandler(Exception)
def _handle_unexpected_error(e: Exception):
    tb = traceback.format_exc()
    app.logger.error(tb)
    if request.path.startswith("/api/") or request.path.startswith("/swagger"):
        return jsonify({"error": str(e), "type": type(e).__name__}), 500
    return Response("Internal Server Error", status=500)

swagger = Swagger(
    app,
    config={
        "specs": [
            {
                "endpoint": "apispec_1",
                "route": "/apispec_1.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "headers": [],
        "specs_route": "/apidocs/",
        "title": "CamAI API Docs",
        "uiversion": 3,
    },
    template={
        "swagger": "2.0",
        "info": {
            "title": "CamAI API",
            "description": "RTSP camera person detection dashboard + API.",
            "version": "1.0.0",
        },
    },
)

@dataclass(frozen=True)
class _QueueJob:
    job_id: str
    batch_id: str
    callback_url: str | None
    camera_id: str
    original_filename: str
    image_bytes: bytes


_job_queue: "queue.Queue[_QueueJob]" = queue.Queue()
_job_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_batches: Dict[str, List[str]] = {}


def _is_http_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _queue_set(job_id: str, payload: Dict[str, Any]) -> None:
    with _job_lock:
        _jobs[job_id] = payload


def _queue_get(job_id: str) -> Dict[str, Any] | None:
    with _job_lock:
        return _jobs.get(job_id)


def _batch_add(batch_id: str, job_id: str) -> None:
    with _job_lock:
        _batches.setdefault(batch_id, []).append(job_id)


def _batch_get(batch_id: str) -> List[str]:
    with _job_lock:
        return list(_batches.get(batch_id, []))


def _post_callback(
    callback_url: str,
    *,
    original_bytes: bytes,
    original_name: str,
    processed_bytes: bytes,
    processed_name: str,
    camera_id: str,
    people_count: int,
    total_people_all: int,
    job_id: str,
    batch_id: str,
) -> None:
    meta = {
        "job_id": job_id,
        "batch_id": batch_id,
        "camera_id": camera_id,
        "people_count": int(people_count),
        "total_people_all": int(total_people_all),
    }
    files = {
        "image": (original_name or "image.jpg", original_bytes, "image/jpeg"),
        "analyzed_image": (processed_name or "analyzed.jpg", processed_bytes, "image/jpeg"),
    }
    data = {
        "meta": json.dumps(meta, ensure_ascii=False),
        "count": str(int(total_people_all)),
    }
    requests.post(callback_url, data=data, files=files, timeout=20)


def _queue_worker() -> None:
    while True:
        job = _job_queue.get()
        try:
            _queue_set(job.job_id, {
                "job_id": job.job_id,
                "batch_id": job.batch_id,
                "status": "processing",
                "camera_id": job.camera_id,
                "callback_url": job.callback_url,
                "original_filename": job.original_filename,
            })

            nparr = np.frombuffer(job.image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image format (cv2.imdecode returned None)")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            uid = uuid.uuid4().hex[:12]
            raw_name = f"queue_{job.camera_id}_{timestamp}_{uid}.jpg"
            processed_name = f"queue_{job.camera_id}_{timestamp}_{uid}_processed.jpg"

            config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            config.UPLOADS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            raw_path = config.UPLOADS_DIR / raw_name
            processed_path = config.UPLOADS_PROCESSED_DIR / processed_name

            cv2.imwrite(str(raw_path), frame)
            people_count, annotated_frame, _ = detect_persons(frame, job.camera_id)
            cv2.imwrite(str(processed_path), annotated_frame)

            ok, processed_buf = cv2.imencode(".jpg", annotated_frame)
            if not ok:
                raise RuntimeError("Failed to encode processed image as jpeg")

            latest = _latest_by_camera()
            cameras_total = sum(int(v.get("people_count", 0) or 0) for v in latest.values())
            total_people_all = cameras_total + int(people_count)

            if job.callback_url:
                _post_callback(
                    job.callback_url,
                    original_bytes=job.image_bytes,
                    original_name=job.original_filename or "image.jpg",
                    processed_bytes=processed_buf.tobytes(),
                    processed_name=processed_name,
                    camera_id=job.camera_id,
                    people_count=int(people_count),
                    total_people_all=int(total_people_all),
                    job_id=job.job_id,
                    batch_id=job.batch_id,
                )

            _queue_set(job.job_id, {
                "job_id": job.job_id,
                "batch_id": job.batch_id,
                "status": "done",
                "camera_id": job.camera_id,
                "callback_url": job.callback_url,
                "people_count": int(people_count),
                "total_people_all": int(total_people_all),
                "raw_image_url": "/outputs/" + str(raw_path.relative_to(config.OUTPUTS_DIR)).replace("\\", "/"),
                "processed_image_url": "/outputs/" + str(processed_path.relative_to(config.OUTPUTS_DIR)).replace("\\", "/"),
            })
        except Exception as e:
            _queue_set(job.job_id, {
                "job_id": job.job_id,
                "batch_id": job.batch_id,
                "status": "failed",
                "camera_id": job.camera_id,
                "callback_url": job.callback_url,
                "error": str(e),
            })
        finally:
            _job_queue.task_done()


_worker_thread = threading.Thread(target=_queue_worker, daemon=True)
_worker_thread.start()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_jsonable(v) for v in obj]
    return str(obj)


@app.route("/swagger.json")
def swagger_json() -> Response:
    with app.app_context():
        spec = swagger.get_apispecs()
    payload = _jsonable(spec)
    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        mimetype="application/json",
    )


@app.route("/swagger")
def swagger_ui() -> Response:
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CamAI Swagger UI</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
    <style>
      body { margin: 0; }
    </style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/swagger.json",
        dom_id: "#swagger-ui",
      });
    </script>
  </body>
</html>
"""
    return Response(html, mimetype="text/html")


def _read_all_stats() -> List[Dict[str, Any]]:
    """Read full stats list from json file."""
    return read_stats_records()


def _write_all_stats(records: List[Dict[str, Any]]) -> None:
    """Write full stats list to json file."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _latest_by_camera() -> Dict[str, Dict[str, Any]]:
    latest = {
        cam_id: {
            "camera_id": cam_id,
            "timestamp": None,
            "people_count": 0,
            "image_path": None,
            "processed_image_path": None,
        }
        for cam_id in config.RTSP_CAMERAS.keys()
    }

    for rec in _read_all_stats():
        cam_id = rec.get("camera_id")
        if cam_id not in latest:
            continue
        prev_ts = latest[cam_id].get("timestamp") or ""
        curr_ts = rec.get("timestamp") or ""
        if curr_ts >= prev_ts:
            latest[cam_id] = rec
    return latest


@app.route("/")
def index():
    return render_template("index.html", cameras=list(config.RTSP_CAMERAS.keys()))


@app.route("/api/status")
def api_status():
    """
    Get latest status summary.
    ---
    tags:
      - status
    responses:
      200:
        description: Status payload
        schema:
          type: object
          properties:
            total_people:
              type: integer
            cameras:
              type: object
            scheduler:
              type: object
    """
    latest = _latest_by_camera()
    total_people = sum(int(v.get("people_count", 0) or 0) for v in latest.values())
    scheduler_state = get_scheduler_state()
    payload = {
        "total_people": total_people,
        "cameras": latest,
        "scheduler": scheduler_state,
    }
    return jsonify(payload)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Upload an image and run detection.
    ---
    tags:
      - upload
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Image file
    responses:
      200:
        description: Detection result
        schema:
          type: object
          properties:
            people_count: { type: integer }
            processed_image_url: { type: string }
            cameras_total_people: { type: integer }
            total_people_all: { type: integer }
      400:
        description: Invalid input
    """
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file"}), 400

    camera_id = "cam1"

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "File bytes are empty"}), 400

    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ext = Path(file.filename).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
        ext = ".jpg"

    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOADS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_name = f"upload_{camera_id}_{timestamp}{ext}"
    processed_name = f"upload_{camera_id}_{timestamp}.jpg"
    raw_path = config.UPLOADS_DIR / raw_name
    processed_path = config.UPLOADS_PROCESSED_DIR / processed_name

    cv2.imwrite(str(raw_path), frame)
    people_count, annotated_frame, _ = detect_persons(frame, camera_id)
    cv2.imwrite(str(processed_path), annotated_frame)

    latest = _latest_by_camera()
    cameras_total = sum(int(v.get("people_count", 0) or 0) for v in latest.values())
    total_people_all = cameras_total + people_count

    processed_url = "/outputs/" + str(processed_path.relative_to(config.OUTPUTS_DIR)).replace("\\", "/")
    return jsonify({
        "people_count": people_count,
        "processed_image_url": processed_url,
        "cameras_total_people": cameras_total,
        "total_people_all": total_people_all,
    })


@app.route("/api/queue/upload", methods=["POST"])
def api_queue_upload():
    """
    Upload one or many JPEG images for queued processing.
    ---
    tags:
      - queue
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: JPEG image file (can be repeated to send a batch)
      - name: callback_url
        in: formData
        type: string
        required: false
        description: Optional. If provided, CamAI will POST results here after each image is processed
    responses:
      202:
        description: Enqueued
      400:
        description: Invalid input
    """
    callback_url = (request.form.get("callback_url") or "").strip() or None
    if callback_url and not _is_http_url(callback_url):
        return jsonify({"error": "callback_url must be a valid http/https URL"}), 400

    camera_id = "cam1"

    files = request.files.getlist("image")
    if not files:
        return jsonify({"error": "No files uploaded (field name: image)"}), 400

    batch_id = uuid.uuid4().hex
    job_ids: List[str] = []

    for f in files:
        if not f or not f.filename:
            continue
        image_bytes = f.read()
        if not image_bytes:
            continue
        job_id = uuid.uuid4().hex
        job_ids.append(job_id)
        _batch_add(batch_id, job_id)
        _queue_set(job_id, {
            "job_id": job_id,
            "batch_id": batch_id,
            "status": "queued",
            "camera_id": camera_id,
            "callback_url": callback_url,
            "original_filename": f.filename,
        })
        _job_queue.put(_QueueJob(
            job_id=job_id,
            batch_id=batch_id,
            callback_url=callback_url,
            camera_id=camera_id,
            original_filename=f.filename,
            image_bytes=image_bytes,
        ))

    if not job_ids:
        return jsonify({"error": "No valid images found in upload"}), 400

    return jsonify({
        "batch_id": batch_id,
        "queued": len(job_ids),
        "job_ids": job_ids,
        "status_url": f"/api/queue/batch/{batch_id}",
    }), 202


@app.route("/api/queue/job/<job_id>", methods=["GET"])
def api_queue_job(job_id: str):
    """
    Get job status/result.
    ---
    tags:
      - queue
    parameters:
      - name: job_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Job payload
      404:
        description: Not found
    """
    job = _queue_get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/queue/batch/<batch_id>", methods=["GET"])
def api_queue_batch(batch_id: str):
    """
    Get batch status + all jobs.
    ---
    tags:
      - queue
    parameters:
      - name: batch_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Batch payload
    """
    try:
        job_ids = _batch_get(batch_id)
        items = [(_queue_get(jid) or {"job_id": jid, "status": "unknown"}) for jid in job_ids]
        return jsonify({
            "batch_id": batch_id,
            "count": len(items),
            "items": items,
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/api/stats", methods=["GET"])
def api_stats_list():
    """
    List all stats records.
    ---
    tags:
      - stats
    responses:
      200:
        description: List of records
        schema:
          type: object
          properties:
            count: { type: integer }
            items:
              type: array
              items: { type: object }
    """
    records = _read_all_stats()
    return jsonify({"count": len(records), "items": records})


@app.route("/api/stats/<int:item_id>", methods=["GET"])
def api_stats_get(item_id: int):
    """
    Get a stats record by index.
    ---
    tags:
      - stats
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Found item
      404:
        description: Item not found
    """
    records = _read_all_stats()
    if item_id < 0 or item_id >= len(records):
        return jsonify({"error": "Item not found"}), 404
    return jsonify({"id": item_id, "item": records[item_id]})


@app.route("/api/stats", methods=["POST"])
def api_stats_create():
    """
    Create a new stats record (append to stats.json).
    ---
    tags:
      - stats
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [timestamp, people_count, image_path, processed_image_path]
          properties:
            timestamp: { type: string }
            people_count: { type: integer }
            image_path: { type: string }
            processed_image_path: { type: string }
    responses:
      201:
        description: Created
      400:
        description: Invalid JSON or missing fields
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON object required"}), 400

    required = ["timestamp", "people_count", "image_path", "processed_image_path"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    records = _read_all_stats()
    records.append(payload)
    _write_all_stats(records)
    return jsonify({"id": len(records) - 1, "item": payload}), 201


@app.route("/api/stats/<int:item_id>", methods=["PUT"])
def api_stats_update(item_id: int):
    """
    Replace a stats record by index.
    ---
    tags:
      - stats
    consumes:
      - application/json
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
      - in: body
        name: body
        required: true
        schema: { type: object }
    responses:
      200:
        description: Updated
      400:
        description: JSON object required
      404:
        description: Item not found
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON object required"}), 400

    records = _read_all_stats()
    if item_id < 0 or item_id >= len(records):
        return jsonify({"error": "Item not found"}), 404

    records[item_id] = payload
    _write_all_stats(records)
    return jsonify({"id": item_id, "item": payload})


@app.route("/api/stats/<int:item_id>", methods=["DELETE"])
def api_stats_delete(item_id: int):
    """
    Delete a stats record by index.
    ---
    tags:
      - stats
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Deleted
      404:
        description: Item not found
    """
    records = _read_all_stats()
    if item_id < 0 or item_id >= len(records):
        return jsonify({"error": "Item not found"}), 404

    deleted = records.pop(item_id)
    _write_all_stats(records)
    return jsonify({"deleted_id": item_id, "deleted_item": deleted, "remaining": len(records)})


@app.route("/outputs/<path:filename>")
def outputs_files(filename: str):
    """
    Serve generated output files.
    ---
    tags:
      - outputs
    parameters:
      - name: filename
        in: path
        type: string
        required: true
    responses:
      200:
        description: File content
      404:
        description: Not found
    """
    return send_from_directory(str(config.OUTPUTS_DIR), filename)
