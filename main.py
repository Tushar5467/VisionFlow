#!/usr/bin/env python3
"""
VisionFlow -- CLI entry point.
==============================
Real-time object detection, multi-object tracking, and video analytics
built on OpenCV's DNN module (YOLOv4-tiny, pre-trained on COCO) plus
classical CV techniques (Canny edges, MOG2 background subtraction,
motion heatmaps). No model is trained here -- everything runs inference
with an off-the-shelf network.

Usage
-----
    python main.py download-model
    python main.py generate-demo
    python main.py run --source data/demo_synthetic.mp4
    python main.py run --source 0                     # webcam
    python main.py run --source path/to/video.mp4 --show
"""

import argparse
import sys

from src import config, utils

logger = utils.get_logger(__name__)


def cmd_download_model(_args) -> int:
    from scripts.download_model import main as download_main
    download_main()
    return 0


def cmd_generate_demo(_args) -> int:
    from scripts.generate_demo_video import generate
    generate()
    return 0


def cmd_run(args) -> int:
    from src.pipeline import VisionFlowPipeline

    source = args.source
    if source.isdigit():
        source = int(source)

    try:
        pipeline = VisionFlowPipeline(
            source=source, save_video=not args.no_save, show_window=args.show
        )
        summary = pipeline.run()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    print("\n=== Run Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nOutputs written to: {config.OUTPUT_DIR}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionflow",
        description="Real-time object detection, tracking & analytics (YOLOv4-tiny + OpenCV).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser(
        "download-model", help="Download the pre-trained YOLOv4-tiny weights/cfg/names."
    )
    p_download.set_defaults(func=cmd_download_model)

    p_demo = subparsers.add_parser(
        "generate-demo", help="Generate a synthetic demo video for smoke-testing."
    )
    p_demo.set_defaults(func=cmd_generate_demo)

    p_run = subparsers.add_parser(
        "run", help="Run the full detection + tracking + analytics pipeline."
    )
    p_run.add_argument(
        "--source", required=True,
        help="Path to a video file, or a webcam index such as 0.",
    )
    p_run.add_argument(
        "--show", action="store_true",
        help="Display a live preview window (requires a GUI-capable environment).",
    )
    p_run.add_argument(
        "--no-save", action="store_true",
        help="Do not write the annotated output video (still writes CSV/JSON/heatmap).",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
