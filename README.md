# VisionFlow — Real-Time Object Detection, Tracking & Video Analytics

A command-line computer vision system that detects, tracks, and counts objects
in a video stream using **YOLOv4-tiny (via OpenCV's DNN module)**, and layers
classical CV analytics — Canny edge detection, MOG2 background subtraction,
and motion heatmaps — on top. Built for **CSE3010 Computer Vision**.

No model is trained anywhere in this project. Detection uses an off-the-shelf,
pre-trained YOLOv4-tiny network; everything else (tracking, analytics,
reporting) is classical, explainable OpenCV/NumPy logic.

---

## Overview

VisionFlow watches a video (a file or a live webcam), and for every frame:

1. **Detects** objects (people, vehicles, animals, everyday objects — 80 COCO
   classes) using YOLOv4-tiny.
2. **Tracks** each detected object across frames with a centroid-distance
   tracker, assigning it a stable ID, and counts objects crossing a
   horizontal line (e.g. "people entering vs. leaving a zone").
3. **Analyzes** the frame with classical CV: Canny edge maps and a
   MOG2-based motion heatmap showing where activity concentrates over time.
4. **Reports** everything to a CSV (per-detection audit trail) and a JSON
   summary (aggregate counts, FPS, elapsed time).

## Features

- Real-time object detection with YOLOv4-tiny (OpenCV DNN backend, CPU-only)
- Multi-object tracking with persistent IDs (centroid-distance association)
- Directional line-crossing counter (in / out)
- Canny edge detection
- MOG2 background subtraction + decaying motion heatmap
- CSV detection log + JSON run summary
- Fully CLI-driven — no GUI required to run or evaluate
- Structured logging to console and file
- Config-driven thresholds (confidence, NMS, tracking distance, etc.)
- 18 automated unit tests (pytest) covering tracker, analytics, detector,
  and report generator

## Technologies / Tools Used

| Component        | Tool |
|-------------------|------|
| Object detection   | YOLOv4-tiny (Darknet weights), loaded via `cv2.dnn` |
| Language           | Python 3.10+ |
| CV library         | OpenCV (`opencv-python`) |
| Numerics           | NumPy |
| Testing            | pytest |
| CLI                | argparse |

## Project Structure

```
visionflow/
├── main.py                        # CLI entry point
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── config.py                  # all tunables in one place
│   ├── utils.py                   # logging, drawing, validation helpers
│   ├── detector.py                # YOLOv4-tiny wrapper (OpenCV DNN)
│   ├── tracker.py                 # CentroidTracker + LineCrossingCounter
│   ├── analytics.py               # EdgeDetector + MotionAnalyzer
│   ├── report_generator.py        # CSV / JSON report writer
│   └── pipeline.py                # orchestrates the full run
├── scripts/
│   ├── download_model.py          # fetches pretrained YOLOv4-tiny files
│   └── generate_demo_video.py     # synthetic video for smoke-testing
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_analytics.py
│   └── test_report_generator.py
├── models/                        # downloaded weights/cfg/names (git-ignored)
├── data/                          # input videos (git-ignored)
├── output/                        # run outputs: video, CSV, JSON, heatmap
└── docs/
    ├── statement.md
    └── diagrams/                  # architecture, workflow, class, use case, sequence
```

## Steps to Install & Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/visionflow.git
cd visionflow
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the pre-trained YOLOv4-tiny model

This fetches `yolov4-tiny.cfg`, `yolov4-tiny.weights` (~23 MB), and
`coco.names` from the official AlexeyAB/darknet GitHub releases into
`models/`. These are third-party, off-the-shelf files — nothing is trained.

```bash
python main.py download-model
```

### 5. Try it immediately with a synthetic demo video (no camera needed)

```bash
python main.py generate-demo
python main.py run --source data/demo_synthetic.mp4
```

### 6. Run on your own video or a webcam

```bash
# Video file
python main.py run --source path/to/your_video.mp4

# Webcam (index 0 is usually the default camera)
python main.py run --source 0

# With a live preview window (needs a GUI-capable environment)
python main.py run --source path/to/video.mp4 --show
```

All runs write to `output/`:

| File | Description |
|---|---|
| `annotated_output.mp4` | Video with bounding boxes, IDs, and the counting line drawn |
| `detections.csv` | One row per detection event: frame, timestamp, object ID, class, confidence, box |
| `summary.json` | Aggregate run statistics: FPS, class counts, line crossings |
| `motion_heatmap.jpg` | Accumulated motion heatmap overlaid on the last frame |
| `edges_sample.jpg` | Canny edge map of the first processed frame |
| `logs/visionflow.log` | Full run log |

### Note for real-world testing

The included `generate-demo` command produces geometric shapes moving on a
blank background — this exercises the full pipeline mechanically, but YOLO
(trained on real-world photos) will not recognise "COCO classes" in it. To see
meaningful detections, run the pipeline on any real video containing people,
vehicles, or everyday objects — for example a phone-recorded clip, a
dashcam clip, or a public sample such as Intel's
[`person-bicycle-car-detection.mp4`](https://github.com/intel-iot-devkit/sample-videos).

## Instructions for Testing

Run the full automated test suite (18 tests, no model download required —
the detector tests use a mocked network so they run offline and fast):

```bash
pip install pytest
pytest tests/ -v
```

Expected result: `18 passed`.

To additionally verify real detection quality (requires the model to be
downloaded first, see step 4 above), run the pipeline against any real photo
or video and inspect `output/`:

```bash
python main.py download-model
python main.py run --source data/demo_synthetic.mp4
cat output/summary.json
```

## Non-Functional Requirements Addressed

- **Performance** — per-inference timing via a `@timeit` decorator; average
  FPS reported in `summary.json`; `PROCESS_EVERY_N_FRAMES` config knob to
  trade accuracy for speed on constrained hardware.
- **Reliability** — graceful handling of missing model files, unreadable
  video sources, and empty detection frames without crashing.
- **Usability** — a single, documented CLI (`main.py`) with `--help` on every
  subcommand; no GUI dependency for grading/execution.
- **Maintainability** — one responsibility per module (`detector`, `tracker`,
  `analytics`, `report_generator`), centralized `config.py`, no hard-coded
  magic numbers in business logic.
- **Logging / Monitoring** — structured logs (console + `output/logs/`) with
  timestamps and levels.
- **Resource efficiency** — CPU-only OpenCV DNN inference on the lightweight
  YOLOv4-tiny variant (not the full YOLOv4), frame-resize option to bound
  per-frame cost.
- **Security** — no data leaves the machine; no network calls at inference
  time (only the one-time `download-model` step fetches public files).

## Design Notes

- **Why YOLOv4-tiny (not full YOLOv4)?** It runs at usable frame rates on
  CPU-only hardware (as used for grading), while still covering all 80 COCO
  classes — an appropriate trade-off for a coursework-scale project.
- **Why a centroid tracker (not a learned re-ID model)?** It's transparent,
  dependency-light, and sufficient for a single-camera, moderate-density
  scene, which keeps the project's complexity proportionate to its scope.
- **Why classical analytics alongside a deep model?** The syllabus
  (Modules 1 and 4) explicitly covers image enhancement, edge detection, and
  background subtraction / motion analysis — including them demonstrates
  those techniques directly rather than relying solely on the detector.

## Screenshots

See `docs/diagrams/` for architecture, workflow, class, use case, and
sequence diagrams, and the project report for sample output frames.

## License

Coursework submission for CSE3010 Computer Vision (VITyarthi). Uses
third-party pre-trained weights (YOLOv4-tiny, AlexeyAB/darknet, GPL-3.0) —
see their repository for licensing of the model weights themselves.
