"""
config.py
---------
Centralised configuration for VisionFlow.

Keeping every tunable value in one module (instead of scattered magic
numbers) is what makes the rest of the codebase maintainable and testable.
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

YOLO_CFG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
COCO_NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

# --------------------------------------------------------------------------- #
# Detection (YOLOv4-tiny via OpenCV DNN)
# --------------------------------------------------------------------------- #
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
SCALE_FACTOR = 1 / 255.0

# Restrict detection to a subset of COCO classes if desired.
# Leave as None to detect every class the model supports.
CLASS_FILTER = None  # e.g. {"person", "car", "bicycle"}

# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
MAX_DISAPPEARED_FRAMES = 15     # frames an object may vanish before it is dropped
MAX_TRACKING_DISTANCE = 75      # px; max centroid distance for the same-object match

# Line-crossing counter: horizontal line, expressed as a fraction of frame height
COUNT_LINE_POSITION = 0.5

# --------------------------------------------------------------------------- #
# Classical CV analytics
# --------------------------------------------------------------------------- #
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

BG_SUBTRACTOR_HISTORY = 300
BG_SUBTRACTOR_VAR_THRESHOLD = 64
BG_SUBTRACTOR_DETECT_SHADOWS = True

HEATMAP_DECAY = 0.98            # multiplicative decay applied to the heatmap each frame
HEATMAP_INTENSITY_INCREMENT = 25

# --------------------------------------------------------------------------- #
# Performance / resource efficiency
# --------------------------------------------------------------------------- #
PROCESS_EVERY_N_FRAMES = 1      # set >1 on constrained hardware to skip frames
FRAME_RESIZE_WIDTH = 640        # None keeps native resolution

# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
CSV_REPORT_NAME = "detections.csv"
JSON_SUMMARY_NAME = "summary.json"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FILE = os.path.join(LOG_DIR, "visionflow.log")
LOG_LEVEL = "INFO"

for _dir in (MODELS_DIR, OUTPUT_DIR, DATA_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)
