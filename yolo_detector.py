# yolo_detector.py
'''
# YOLOv8-based person detection with higher confidence threshold.
# This module uses the YOLOv8 model to detect persons in a video stream.
# It applies a higher confidence threshold to filter out low-confidence detections.
# It also includes gamma correction for image brightness adjustment and an advanced box filter to remove small or non-standard aspect ratio boxes.  '''
import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLOv8-based person detection with higher confidence threshold.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        conf_thres: float = 0.7,      
        iou_thres: float = 0.45,
        person_class_id: int = 0,
        gamma: float = 1.0,
        advanced_box_filter: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.overrides["conf"] = conf_thres
        self.model.overrides["iou"] = iou_thres

        self.person_class_id = person_class_id
        self.gamma = gamma
        self.advanced_box_filter = advanced_box_filter
        self.cap = None

        logger.info("YOLODetector with model=%s, device=%s, conf=%.2f", model_path, device, conf_thres)

    def open_camera(self, camera_index: int = 0) -> bool:
        logger.info("Opening camera at index %d", camera_index)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera index %d", camera_index)
            return False
        return True

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera not opened.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return None

        if self.gamma != 1.0:
            frame = self.apply_gamma_correction(frame, self.gamma)
        return frame

    def detect(self, frame: np.ndarray):
        results = self.model(frame, stream=False)[0]
        boxes = []
        confidences = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id == self.person_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if self.advanced_box_filter and not self.box_passes_filter(x1, y1, x2, y2, frame.shape):
                    continue
                boxes.append((x1, y1, x2, y2))
                confidences.append(conf)

        return boxes, confidences

    def close_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    @staticmethod
    def apply_gamma_correction(frame: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def box_passes_filter(self, x1, y1, x2, y2, shape):
        h, w, _ = shape
        box_w = x2 - x1
        box_h = y2 - y1

        min_area = 80 * 80
        if box_w * box_h < min_area:
            return False

        aspect_ratio = box_w / float(box_h + 1e-6)
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            return False

        return True
