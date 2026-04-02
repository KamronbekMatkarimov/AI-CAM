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


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = float(area_a + area_b - inter)
    if denom <= 0:
        return 0.0
    return float(inter) / denom


def _dedupe_by_iou(dets: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    if not dets:
        return dets
    if iou_thr is None:
        return dets
    try:
        thr = float(iou_thr)
    except Exception:
        return dets

    if thr <= 0:
        return dets

    # Greedy NMS on already-NMSed boxes: removes rare duplicates YOLO can still leave.
    dets_sorted = sorted(dets, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    kept_boxes: List[Tuple[int, int, int, int]] = []
    for d in dets_sorted:
        bb = d.get("bbox")
        if not bb or len(bb) != 4:
            continue
        box = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        if any(_iou_xyxy(box, kb) >= thr for kb in kept_boxes):
            continue
        kept.append(d)
        kept_boxes.append(box)
    return kept


def detect_persons(
    frame: np.ndarray,
    camera_id: str,
) -> Tuple[int, np.ndarray, List[Dict[str, Any]]]:
    model = _get_model()

    settings = config.CAMERA_DETECTION_SETTINGS.get(
        camera_id,
        {"imgsz": 1280, "conf": 0.15},
    )
    imgsz = int(settings.get("imgsz", 1280))
    conf = float(settings.get("conf", 0.15))
    iou = float(settings.get("iou", config.IOU_THRESHOLD))
    augment = bool(settings.get("augment", config.YOLO_AUGMENT))
    max_det = int(settings.get("max_det", config.YOLO_MAX_DET))
    min_area_ratio = float(settings.get("min_box_area_ratio", config.MIN_BOX_AREA_RATIO))
    min_h_ratio = float(settings.get("min_box_height_ratio", config.MIN_BOX_HEIGHT_RATIO))
    dedupe_iou = settings.get("dedupe_iou", 0.65)

    # IMPORTANT: run YOLO on the original frame. Pre-resizing here causes double-resize
    # (our resize + YOLO's letterbox), which hurts small-person recall and stability.
    proc = frame
    proc_h, proc_w = proc.shape[:2]
    results = model.predict(
        proc,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[config.PERSON_CLASS_ID],
        augment=augment,
        max_det=max_det,
        verbose=False,
    )

    detections = []
    annotated = proc.copy()
    frame_area = float(proc_h * proc_w)

    # Ultralytics already applies conf threshold and NMS; we only do size sanity filters here.
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])
        cls_id = int(box.cls[0])

        if cls_id != config.PERSON_CLASS_ID:
            continue

        x1 = max(0, min(x1, proc_w - 1))
        y1 = max(0, min(y1, proc_h - 1))
        x2 = max(0, min(x2, proc_w))
        y2 = max(0, min(y2, proc_h))

        if x2 <= x1 or y2 <= y1:
            continue

        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if frame_area > 0 and (box_area / frame_area) < min_area_ratio:
            continue

        box_h = float(y2 - y1)
        if proc_h > 0 and (box_h / float(proc_h)) < min_h_ratio:
            continue

        crop = proc[y1:y2, x1:x2].copy()

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf_val,
            "class_id": cls_id,
            "crop": crop,
        })

    detections = _dedupe_by_iou(detections, dedupe_iou)

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        conf_val = float(d.get("confidence", 0.0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"person {conf_val:.2f}",
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
