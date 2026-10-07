"""Unit tests for src.detector.YOLODetector.

These tests avoid requiring the actual (large) YOLO weight files to be
present in CI: `test_missing_model_files_raise_clear_error` checks the
validation path, and `test_detect_postprocessing_with_stubbed_network`
monkeypatches cv2.dnn so the NMS/parsing logic is verified without a
real forward pass. An integration-style manual check against the real
model is documented in the README.
"""

import os

import numpy as np
import pytest

from src.detector import YOLODetector, Detection


def test_missing_model_files_raise_clear_error(tmp_path):
    fake_cfg = str(tmp_path / "missing.cfg")
    fake_weights = str(tmp_path / "missing.weights")
    fake_names = str(tmp_path / "missing.names")

    with pytest.raises(FileNotFoundError):
        YOLODetector(cfg_path=fake_cfg, weights_path=fake_weights, names_path=fake_names)


class _StubNet:
    """Mimics the subset of cv2.dnn.Net's API that YOLODetector calls."""

    def __init__(self, fake_output):
        self._fake_output = fake_output

    def setPreferableBackend(self, *_):
        pass

    def setPreferableTarget(self, *_):
        pass

    def getLayerNames(self):
        return ["layer1", "layer2", "yolo_out"]

    def getUnconnectedOutLayers(self):
        return np.array([3])

    def setInput(self, *_):
        pass

    def forward(self, *_):
        return [self._fake_output]


def _make_fake_output(class_id, confidence, cx, cy, w, h, num_classes=80):
    """Builds one YOLO-style output row: [cx, cy, w, h, obj_conf, class_scores...]."""
    row = np.zeros(5 + num_classes, dtype=np.float32)
    row[0:4] = [cx, cy, w, h]
    row[4] = confidence  # objectness (unused directly, kept for shape realism)
    scores = np.zeros(num_classes, dtype=np.float32)
    scores[class_id] = confidence
    row[5:] = scores
    return np.array([row])


@pytest.fixture
def detector_with_stub(tmp_path, monkeypatch):
    cfg = tmp_path / "yolov4-tiny.cfg"
    weights = tmp_path / "yolov4-tiny.weights"
    names = tmp_path / "coco.names"
    cfg.write_text("stub")
    weights.write_text("stub")
    names.write_text("\n".join(["person", "car"] + [f"class{i}" for i in range(2, 80)]))

    fake_output = _make_fake_output(class_id=0, confidence=0.9, cx=0.5, cy=0.5, w=0.2, h=0.3)

    monkeypatch.setattr("src.detector._load_network", lambda *a, **k: _StubNet(fake_output))
    monkeypatch.setattr(
        "src.detector.cv2.dnn.blobFromImage",
        lambda *a, **k: np.zeros((1, 3, 416, 416), dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.detector.cv2.dnn.NMSBoxes",
        lambda boxes, confidences, conf_th, nms_th: list(range(len(boxes))),
    )

    return YOLODetector(cfg_path=str(cfg), weights_path=str(weights), names_path=str(names))


def test_detect_returns_expected_detection(detector_with_stub):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector_with_stub.detect(frame)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.9)
    assert len(det.box) == 4


def test_detect_below_confidence_threshold_is_filtered(tmp_path, monkeypatch):
    cfg = tmp_path / "yolov4-tiny.cfg"
    weights = tmp_path / "yolov4-tiny.weights"
    names = tmp_path / "coco.names"
    cfg.write_text("stub")
    weights.write_text("stub")
    names.write_text("\n".join(["person"] + [f"class{i}" for i in range(1, 80)]))

    low_conf_output = _make_fake_output(class_id=0, confidence=0.1, cx=0.5, cy=0.5, w=0.2, h=0.3)

    monkeypatch.setattr("src.detector._load_network", lambda *a, **k: _StubNet(low_conf_output))
    monkeypatch.setattr(
        "src.detector.cv2.dnn.blobFromImage",
        lambda *a, **k: np.zeros((1, 3, 416, 416), dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.detector.cv2.dnn.NMSBoxes",
        lambda boxes, confidences, conf_th, nms_th: list(range(len(boxes))),
    )

    detector = YOLODetector(
        cfg_path=str(cfg), weights_path=str(weights), names_path=str(names),
        confidence_threshold=0.5,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []
