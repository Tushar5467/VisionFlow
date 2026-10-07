"""
detector.py
-----------
Thin, well-tested wrapper around OpenCV's DNN module loading a
pre-trained YOLOv4-tiny model (Darknet weights + cfg, trained on COCO).

No model is trained or fine-tuned here -- this module only performs
inference with an off-the-shelf network, exactly as required.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src import config, utils

logger = utils.get_logger(__name__)


def _load_network(cfg_path: str, weights_path: str):
    darknet_loader = getattr(cv2.dnn, "readNetFromDarknet", None)
    if darknet_loader is not None:
        return darknet_loader(cfg_path, weights_path)
    raise RuntimeError(
        "This OpenCV build does not support Darknet models. "
        "Install dependencies again to use opencv-python 4.x."
    )


@dataclass
class Detection:
    box: tuple          # (x, y, w, h) in pixel coordinates
    class_id: int
    class_name: str
    confidence: float


class YOLODetector:
    """Loads YOLOv4-tiny once and exposes a simple `.detect(frame)` API."""

    def __init__(
        self,
        cfg_path: str = config.YOLO_CFG_PATH,
        weights_path: str = config.YOLO_WEIGHTS_PATH,
        names_path: str = config.COCO_NAMES_PATH,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.NMS_THRESHOLD,
        class_filter=config.CLASS_FILTER,
    ):
        utils.validate_file_exists(
            cfg_path, "Run `python main.py download-model` to fetch it."
        )
        utils.validate_file_exists(
            weights_path, "Run `python main.py download-model` to fetch it."
        )

        self.class_names = utils.load_class_names(names_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.class_filter = class_filter

        logger.info("Loading YOLOv4-tiny network via OpenCV DNN ...")
        self.net = _load_network(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        # OpenCV versions differ in whether this is a flat array or nested.
        self.output_layers = [
            layer_names[i - 1] if isinstance(i, (int, np.integer)) else layer_names[i[0] - 1]
            for i in unconnected
        ]
        logger.info(f"Model loaded. {len(self.class_names)} classes available.")

    @utils.timeit()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run one forward pass and return post-NMS detections."""
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            config.SCALE_FACTOR,
            (config.INPUT_WIDTH, config.INPUT_HEIGHT),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.confidence_threshold:
                    continue

                class_name = self.class_names[class_id]
                if self.class_filter and class_name not in self.class_filter:
                    continue

                cx, cy, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )

        detections = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                detections.append(
                    Detection(
                        box=tuple(boxes[i]),
                        class_id=class_ids[i],
                        class_name=self.class_names[class_ids[i]],
                        confidence=confidences[i],
                    )
                )
        return detections
