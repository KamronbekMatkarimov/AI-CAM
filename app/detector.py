import logging
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        logger.info("Loading YOLO model: %s", config.YOLO_MODEL)
        _model = YOLO(config.YOLO_MODEL)
    return _model


def _resize_for_detection(frame: np.ndarray, target_width: int) -> Tuple[np.ndarray, float]:
    orig_h, orig_w = frame.shape[:2]
    if orig_w <= 0 or target_width <= 0:
        return frame, 1.0
    if orig_w == target_width:
        return frame, 1.0

    scale = float(target_width) / float(orig_w)
    new_h = max(1, int(orig_h * scale))
    resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def detect_persons(
    frame: np.ndarray,
    camera_id: str,
) -> Tuple[int, np.ndarray, List[Dict[str, Any]]]:
    model = _get_model()

    settings = config.CAMERA_DETECTION_SETTINGS.get(
        camera_id,
        {"imgsz": 1280, "conf": 0.15},
    )
    target_width = int(settings["imgsz"]) 
    conf = float(settings["conf"])

    resized, scale = _resize_for_detection(frame, target_width)

    resized_h, resized_w = resized.shape[:2]
    results = model.predict(
        resized,
        conf=conf,
        iou=config.IOU_THRESHOLD,
        imgsz=target_width,
        classes=[config.PERSON_CLASS_ID],
        augment=config.YOLO_AUGMENT,
        max_det=config.YOLO_MAX_DET,
        verbose=False,
    )

    people = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf_val = float(box.conf[0])

        if cls == config.PERSON_CLASS_ID:
            if conf_val < conf:
                continue
            people.append(box)

    detections = []
    annotated = resized.copy()
    proc_h, proc_w = resized.shape[:2]
    frame_area = float(proc_h * proc_w)

    for box in people:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])
        cls_id = int(box.cls[0])

        x1 = max(0, min(x1, proc_w - 1))
        y1 = max(0, min(y1, proc_h - 1))
        x2 = max(0, min(x2, proc_w))
        y2 = max(0, min(y2, proc_h))

        if x2 <= x1 or y2 <= y1:
            continue

        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if frame_area > 0 and (box_area / frame_area) < config.MIN_BOX_AREA_RATIO:
            continue

        box_h = float(y2 - y1)
        if proc_h > 0 and (box_h / float(proc_h)) < config.MIN_BOX_HEIGHT_RATIO:
            continue

        crop = resized[y1:y2, x1:x2].copy()

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf_val,
            "class_id": cls_id,
            "crop": crop,
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "person",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    people_count = len(detections)

    cv2.putText(
        annotated,
        f"Total: {people_count} people",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )

    if detections:
        classes_str = ", ".join(f"person({d['confidence']:.2f})" for d in detections)
        logger.debug("%s detected classes: %s", camera_id, classes_str)
        logger.info("%s: %d person(s) - confs: %s", camera_id, people_count, [f"{d['confidence']:.2f}" for d in detections])
    else:
        logger.debug("%s detected classes: none", camera_id)
        logger.info("%s: 0 persons", camera_id)

    return people_count, annotated, detections
