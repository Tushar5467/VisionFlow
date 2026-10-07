"""
generate_demo_video.py
-----------------------
Generates a short synthetic .mp4 with moving coloured shapes, so the
pipeline can be smoke-tested end-to-end without needing a webcam or a
real-world video file on hand. (For meaningful YOLO detections, run the
pipeline on a real video containing COCO-class objects such as people or
vehicles -- see README for a sample.)

Run:
    python scripts/generate_demo_video.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, utils  # noqa: E402

logger = utils.get_logger(__name__)


def generate(
    output_path: str = None,
    width: int = 640,
    height: int = 480,
    fps: int = 20,
    duration_seconds: int = 6,
) -> str:
    output_path = output_path or os.path.join(config.DATA_DIR, "demo_synthetic.mp4")
    utils.ensure_dir(os.path.dirname(output_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    n_frames = fps * duration_seconds
    rng = np.random.default_rng(42)

    # Two "objects" bouncing across the frame, rectangle + circle shapes,
    # large and high-contrast enough to also exercise Canny / MOG2 cleanly.
    box_pos = np.array([50, 100], dtype=float)
    box_vel = np.array([6, 3], dtype=float)
    circ_pos = np.array([500, 300], dtype=float)
    circ_vel = np.array([-5, 4], dtype=float)

    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)

        box_pos += box_vel
        if not (0 < box_pos[0] < width - 80):
            box_vel[0] *= -1
        if not (0 < box_pos[1] < height - 60):
            box_vel[1] *= -1

        circ_pos += circ_vel
        if not (30 < circ_pos[0] < width - 30):
            circ_vel[0] *= -1
        if not (30 < circ_pos[1] < height - 30):
            circ_vel[1] *= -1

        cv2.rectangle(
            frame,
            (int(box_pos[0]), int(box_pos[1])),
            (int(box_pos[0]) + 80, int(box_pos[1]) + 60),
            (0, 140, 255),
            -1,
        )
        cv2.circle(frame, (int(circ_pos[0]), int(circ_pos[1])), 30, (255, 100, 0), -1)
        cv2.putText(
            frame, f"frame {i}", (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic demo video written to {output_path} ({n_frames} frames)")
    return output_path


if __name__ == "__main__":
    generate()
