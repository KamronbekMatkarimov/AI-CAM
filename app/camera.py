"""
RTSP camera capture module.
Connects to camera, captures frame, releases.
Protection from corrupted/bad frames (RTSP/Hikvision issue).
"""

import time
import cv2
import logging
from typing import Optional

import numpy as np
import config

logger = logging.getLogger(__name__)

MIN_FRAME_MEAN = 5
WARMUP_FRAMES = 5


def capture_frame(
    rtsp_url: str,
    timeout_ms: int = None,
    retry_attempts: int = None,
) -> Optional[np.ndarray]:
    timeout_ms = timeout_ms or (config.RTSP_TIMEOUT_SECONDS * 1000)
    retry_attempts = retry_attempts or config.RTSP_RETRY_ATTEMPTS

    for attempt in range(1, retry_attempts + 1):
        cap = None
        try:
            logger.info("Connecting to RTSP (attempt %d/%d)", attempt, retry_attempts)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logger.warning("Failed to open RTSP stream (attempt %d)", attempt)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

            time.sleep(1)

            frame = None
            for _ in range(WARMUP_FRAMES):
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f

            if frame is None:
                logger.warning("Failed to read frame (attempt %d)", attempt)
                continue

            if frame.mean() < MIN_FRAME_MEAN:
                logger.warning("Bad frame (mean=%.1f < %d) - black/corrupted, skipping", frame.mean(), MIN_FRAME_MEAN)
                continue

            logger.info("Frame captured successfully (mean=%.1f)", frame.mean())
            return frame

        except Exception as e:
            logger.error("Capture error (attempt %d): %s", attempt, str(e))

        finally:
            if cap is not None:
                cap.release()
                logger.debug("Camera released")

    logger.error("All %d capture attempts failed", retry_attempts)
    return None
