"""
tracker.py
----------
A lightweight centroid-based multi-object tracker, plus a line-crossing
counter. This is classical, explainable tracking (no re-identification
network) -- it associates detections frame-to-frame purely by nearest
centroid distance, which is enough for a single-camera, low-clutter feed
and keeps the whole system dependency-light (OpenCV + numpy only).
"""

from collections import OrderedDict
from typing import Dict, List

import numpy as np

from src import config, utils

logger = utils.get_logger(__name__)


class CentroidTracker:
    def __init__(
        self,
        max_disappeared: int = config.MAX_DISAPPEARED_FRAMES,
        max_distance: int = config.MAX_TRACKING_DISTANCE,
    ):
        self.next_object_id = 0
        self.objects: Dict[int, tuple] = OrderedDict()       # id -> centroid
        self.boxes: Dict[int, tuple] = OrderedDict()          # id -> (x, y, w, h)
        self.disappeared: Dict[int, int] = OrderedDict()      # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid, box):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]

    def update(self, detections: List) -> Dict[int, tuple]:
        """Associate the current frame's detections with tracked objects.

        `detections` is a list of Detection objects (see detector.py) or
        any object exposing a `.box` attribute of (x, y, w, h).
        Returns a mapping of object_id -> box for the current frame.
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.boxes

        input_centroids = [utils.centroid_of_box(d.box) for d in detections]
        input_boxes = [d.box for d in detections]

        if len(self.objects) == 0:
            for centroid, box in zip(input_centroids, input_boxes):
                self._register(centroid, box)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
                axis=2,
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = input_boxes[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            for col in unused_cols:
                self._register(input_centroids[col], input_boxes[col])

        return self.boxes


class LineCrossingCounter:
    """Counts objects crossing a horizontal line, split by direction."""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.count_down = 0   # crossed top -> bottom
        self.count_up = 0     # crossed bottom -> top
        self._last_y: Dict[int, int] = {}
        self._counted: set = set()

    def update(self, tracked_boxes: Dict[int, tuple]) -> None:
        for object_id, box in tracked_boxes.items():
            cy = utils.centroid_of_box(box)[1]
            prev_y = self._last_y.get(object_id)

            if prev_y is not None and object_id not in self._counted:
                crossed_down = prev_y < self.line_y <= cy
                crossed_up = prev_y > self.line_y >= cy
                if crossed_down:
                    self.count_down += 1
                    self._counted.add(object_id)
                elif crossed_up:
                    self.count_up += 1
                    self._counted.add(object_id)

            self._last_y[object_id] = cy

    @property
    def total(self) -> int:
        return self.count_down + self.count_up
