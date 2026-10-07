"""
download_model.py
------------------
Fetches the pre-trained YOLOv4-tiny weights, config, and COCO class
names into models/. These are third-party, off-the-shelf files -- this
project does not train or fine-tune any model.

Run standalone:
    python scripts/download_model.py

Or via the CLI:
    python main.py download-model
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, utils  # noqa: E402

logger = utils.get_logger(__name__)

FILES = {
    config.YOLO_CFG_PATH: (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    ),
    config.YOLO_WEIGHTS_PATH: (
        "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
    ),
    config.COCO_NAMES_PATH: (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    ),
}


def _download(url: str, dest: str) -> None:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        logger.info(f"Already present, skipping: {dest}")
        return

    logger.info(f"Downloading {url} -> {dest}")
    tmp_path = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest)
        logger.info(f"Saved {dest} ({os.path.getsize(dest) / 1024:.1f} KB)")
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def main() -> None:
    utils.ensure_dir(config.MODELS_DIR)
    for dest, url in FILES.items():
        _download(url, dest)
    logger.info("All model files are ready in the models/ directory.")


if __name__ == "__main__":
    main()
