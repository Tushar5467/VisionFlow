"""
report_generator.py
--------------------
Turns raw per-frame detection/tracking events into two artefacts:

  * detections.csv -- one row per detection event (frame-level audit trail)
  * summary.json    -- run-level aggregate statistics

Kept separate from the detection/tracking logic so the storage format can
change without touching the CV pipeline (single-responsibility).
"""

import csv
import json
import os
import time
from collections import Counter
from datetime import datetime

from src import config, utils

logger = utils.get_logger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        self.output_dir = output_dir
        utils.ensure_dir(self.output_dir)

        self.csv_path = os.path.join(self.output_dir, config.CSV_REPORT_NAME)
        self.json_path = os.path.join(self.output_dir, config.JSON_SUMMARY_NAME)

        self._rows = []
        self._class_counts = Counter()
        self._start_time = time.time()
        self._frame_count = 0

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["frame", "timestamp", "object_id", "class_name", "confidence", "x", "y", "w", "h"]
        )

    def log_frame(self, frame_index: int, tracked_detections: list) -> None:
        """`tracked_detections` items: (object_id, class_name, confidence, box)."""
        self._frame_count = frame_index
        ts = round(time.time() - self._start_time, 3)

        for object_id, class_name, confidence, box in tracked_detections:
            x, y, w, h = box
            self._csv_writer.writerow(
                [frame_index, ts, object_id, class_name, round(confidence, 4), x, y, w, h]
            )
            self._class_counts[class_name] += 1

    def finalize(self, extra_stats: dict = None) -> dict:
        """Flush CSV and write the JSON summary. Returns the summary dict."""
        self._csv_file.close()

        elapsed = round(time.time() - self._start_time, 2)
        avg_fps = round(self._frame_count / elapsed, 2) if elapsed > 0 else 0.0

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "frames_processed": self._frame_count,
            "elapsed_seconds": elapsed,
            "average_fps": avg_fps,
            "detections_by_class": dict(self._class_counts),
            "total_detection_events": int(sum(self._class_counts.values())),
        }
        if extra_stats:
            summary.update(extra_stats)

        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Report written: {self.csv_path}")
        logger.info(f"Summary written: {self.json_path}")
        return summary
