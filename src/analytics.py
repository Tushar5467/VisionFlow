"""
analytics.py
------------
Classical computer-vision analytics that run alongside the YOLO detector:

1. Canny edge detection        -> structural / boundary information
2. MOG2 background subtraction -> foreground motion mask
3. Motion heatmap accumulation -> where activity concentrates over time

These techniques come directly from the course syllabus (Module 1: image
enhancement / filtering; Module 4: background subtraction & motion
analysis) and complement the deep-learning detector with transparent,
non-learned signals.
"""

import cv2
import numpy as np

from src import config, utils

logger = utils.get_logger(__name__)


class EdgeDetector:
    def __init__(self, low=config.CANNY_LOW_THRESHOLD, high=config.CANNY_HIGH_THRESHOLD):
        self.low = low
        self.high = high

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, self.low, self.high)


class MotionAnalyzer:
    """Wraps MOG2 background subtraction and a decaying motion heatmap."""

    def __init__(self, frame_shape=None):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.BG_SUBTRACTOR_HISTORY,
            varThreshold=config.BG_SUBTRACTOR_VAR_THRESHOLD,
            detectShadows=config.BG_SUBTRACTOR_DETECT_SHADOWS,
        )
        self.heatmap = None
        if frame_shape is not None:
            self._init_heatmap(frame_shape)

    def _init_heatmap(self, frame_shape):
        h, w = frame_shape[:2]
        self.heatmap = np.zeros((h, w), dtype=np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return the raw foreground mask and update the heatmap in place."""
        if self.heatmap is None:
            self._init_heatmap(frame.shape)

        fg_mask = self.bg_subtractor.apply(frame)
        # Shadows are labelled 127 by MOG2; keep only confident foreground (255).
        fg_mask_binary = (fg_mask == 255).astype(np.uint8) * 255

        self.heatmap *= config.HEATMAP_DECAY
        self.heatmap += (fg_mask_binary / 255.0) * config.HEATMAP_INTENSITY_INCREMENT
        np.clip(self.heatmap, 0, 255, out=self.heatmap)

        return fg_mask_binary

    def get_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        if self.heatmap is None:
            return frame.copy()
        normalized = self.heatmap.astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
