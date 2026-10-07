"""
utils.py
--------
Shared helpers: logging setup, timing decorator, drawing helpers, and
small validation utilities used across the pipeline.
"""

import functools
import logging
import os
import time

import cv2
import numpy as np

from src import config


def get_logger(name: str = "visionflow") -> logging.Logger:
    """Return a module-level logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def timeit(logger: logging.Logger = None):
    """Decorator that logs the wall-clock execution time of a function.

    Used to satisfy the 'performance measurement' non-functional requirement.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger()
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _logger.debug(f"{func.__name__} took {elapsed_ms:.2f} ms")
            return result

        return wrapper

    return decorator


def validate_file_exists(path: str, hint: str = "") -> None:
    """Raise a clear, actionable error if a required file is missing."""
    if not os.path.isfile(path):
        message = f"Required file not found: {path}"
        if hint:
            message += f"\n  -> {hint}"
        raise FileNotFoundError(message)


def load_class_names(path: str) -> list:
    validate_file_exists(
        path, hint="Run `python main.py download-model` first."
    )
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def draw_box(frame: np.ndarray, box, label: str, color=(0, 255, 0)) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
    cv2.putText(
        frame, label, (x + 2, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def resize_frame(frame: np.ndarray, target_width: int = None) -> np.ndarray:
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, int(h * scale)))


def centroid_of_box(box) -> tuple:
    x, y, w, h = box
    return (int(x + w / 2.0), int(y + h / 2.0))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
