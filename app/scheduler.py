import logging
import time
from datetime import datetime, timedelta

import cv2

import config
from app.camera import capture_frame
from app.detector import detect_persons
from app.utils import append_stats_record, ensure_directories, get_timestamp_str

logger = logging.getLogger(__name__)

_scheduler_state = {
    "interval_seconds": config.INTERVAL_SECONDS,
    "last_run_ts": None,
    "next_run_ts": None,
}


def get_scheduler_state():
    return dict(_scheduler_state)


def process_camera(camera_id: str, rtsp_url: str) -> None:
    timestamp_str = get_timestamp_str()
    raw_filename = f"{camera_id}_{timestamp_str}.jpg"
    processed_filename = f"{camera_id}_{timestamp_str}.jpg"

    raw_path = config.RAW_IMAGES_DIR / raw_filename
    processed_path = config.PROCESSED_IMAGES_DIR / processed_filename

    frame = capture_frame(rtsp_url)
    if frame is None:
        logger.warning("%s: Bad frame, skipping...", camera_id)
        print(f"  [{camera_id}] ⚠️ Bad frame, skipping...")
        return

    try:
        cv2.imwrite(str(raw_path), frame)
        logger.info("Saved raw image: %s", raw_path)
    except Exception as e:
        logger.error("Failed to save raw image: %s", e)
        return

    try:
        people_count, annotated_frame, detections = detect_persons(frame, camera_id)
    except Exception as e:
        logger.error("Detection failed for %s: %s", camera_id, e)
        people_count = 0
        annotated_frame = frame
        detections = []

    try:
        cv2.imwrite(str(processed_path), annotated_frame)
        logger.info("Saved processed image: %s", processed_path)
    except Exception as e:
        logger.error("Failed to save processed image: %s", e)

    for i, det in enumerate(detections, start=1):
        crop = det.get("crop")
        if crop is not None and crop.size > 0:
            crop_filename = f"{camera_id}_{timestamp_str}_person_{i}.jpg"
            crop_path = config.CROPS_DIR / crop_filename
            try:
                cv2.imwrite(str(crop_path), crop)
                logger.debug("Saved crop: %s", crop_path)
            except Exception as e:
                logger.warning("Failed to save crop %s: %s", crop_filename, e)

    record = {
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat(),
        "people_count": people_count,
        "image_path": str(raw_path),
        "processed_image_path": str(processed_path),
    }
    append_stats_record(record)

    print(f"  [{camera_id}] People detected: {people_count}")


def run_cycle() -> None:
    """Run one capture cycle for all cameras."""
    logger.info("Starting capture cycle at %s", datetime.now().isoformat())
    _scheduler_state["last_run_ts"] = datetime.now().isoformat()
    print("\n--- Capture cycle ---")
    ensure_directories()

    for camera_id, rtsp_url in config.RTSP_CAMERAS.items():
        try:
            process_camera(camera_id, rtsp_url)
        except Exception as e:
            logger.exception("Error processing %s: %s", camera_id, e)
            print(f"  [{camera_id}] Error: {e}")
        time.sleep(2)

    logger.info("Capture cycle completed")
    print("--- Cycle complete ---\n")


def run_scheduler(interval_seconds: int = None) -> None:
    interval = int(interval_seconds or config.INTERVAL_SECONDS)
    _scheduler_state["interval_seconds"] = interval

    logger.info("Scheduler started. Interval: %d seconds", interval)
    print(f"CamAI started. Capture every {interval // 60} minutes.\n")

    next_run_time = time.monotonic()
    while True:
        now = time.monotonic()
        sleep_s = next_run_time - now
        if sleep_s > 0:
            _scheduler_state["next_run_ts"] = (datetime.now() + timedelta(seconds=sleep_s)).isoformat()
            time.sleep(sleep_s)

        _scheduler_state["last_run_ts"] = datetime.now().isoformat()
        run_cycle()

        next_run_time = next_run_time + interval

        remaining = max(0.0, next_run_time - time.monotonic())
        _scheduler_state["next_run_ts"] = (datetime.now() + timedelta(seconds=remaining)).isoformat()
