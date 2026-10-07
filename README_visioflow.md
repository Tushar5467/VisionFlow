# VisionFlow

Object detection, tracking and video analytics from the command line, using YOLOv4-tiny and OpenCV.

- **Course:** CSE3010 Computer Vision, VIT Bhopal University (AY 2026-27)
- **Student:** Tushar Chakraborty
- **Registration No:** 24BAI10842

VisionFlow is a terminal-based tool that finds objects in a video, follows them from frame to frame, and counts how many cross a line. The detection is done by a pre-trained YOLOv4-tiny network loaded through OpenCV's DNN module. Canny edge detection, MOG2 background subtraction and a motion heatmap are added on top of that. I built it as my project for CSE3010 Computer Vision.

Nothing in this project is trained. The detector is an off-the-shelf network, and the tracking, analytics and reporting are plain OpenCV and NumPy code that can be read and explained line by line.

---

## What it does

Give VisionFlow a video file or a webcam and it handles each frame in four steps:

1. **Detect.** YOLOv4-tiny looks for objects from the 80 COCO classes (people, vehicles, animals, common everyday items).
2. **Track.** A centroid-distance tracker gives every detected object an ID that stays the same between frames. A horizontal counting line records how many objects go in each direction, for example people entering or leaving an area.
3. **Analyse.** The frame is run through Canny edge detection, and MOG2 background subtraction feeds a heatmap showing where movement piled up over time.
4. **Report.** Every detection is written to a CSV file, and a JSON file stores the totals for the run (class counts, FPS, time taken).

## Features

- Detection with YOLOv4-tiny on the OpenCV DNN backend, running on CPU only
- Object tracking with IDs that persist across frames
- In and out line-crossing counts
- Canny edge maps
- MOG2 foreground extraction with a motion heatmap that fades over time
- CSV log of detections and a JSON run summary
- Works entirely from the command line, no GUI needed
- Logging to both the console and a file
- Thresholds (confidence, NMS, tracking distance and so on) set in one config file
- 18 pytest unit tests for the tracker, analytics, detector and report generator

## Tools used

| Part | Tool |
|---|---|
| Detector | YOLOv4-tiny (Darknet weights) loaded with `cv2.dnn` |
| Language | Python 3.10 or newer |
| Computer vision | OpenCV (`opencv-python`) |
| Numerical work | NumPy |
| Tests | pytest |
| Command line | argparse |

## Project layout

```
VisionFlow/
├── main.py                        # command-line entry point
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── config.py                  # all adjustable settings
│   ├── utils.py                   # logging, drawing and validation helpers
│   ├── detector.py                # YOLOv4-tiny wrapper using OpenCV DNN
│   ├── tracker.py                 # CentroidTracker and LineCrossingCounter
│   ├── analytics.py               # EdgeDetector and MotionAnalyzer
│   ├── report_generator.py        # writes the CSV and JSON reports
│   └── pipeline.py                # runs everything together
├── scripts/
│   ├── download_model.py          # downloads the YOLOv4-tiny files
│   └── generate_demo_video.py     # makes a synthetic test video
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_analytics.py
│   └── test_report_generator.py
├── models/                        # weights, cfg and class names (git-ignored)
├── data/                          # input videos (git-ignored)
├── output/                        # results: video, CSV, JSON, heatmap
└── docs/
    ├── statement.md
    └── diagrams/                  # architecture, workflow, class, use case, sequence
```

## Setup and usage

### 1. Get the code

```bash
git clone https://github.com/Tushar5467/VisionFlow.git
cd VisionFlow
```

### 2. Make a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Download the YOLOv4-tiny files

This saves `yolov4-tiny.cfg`, `yolov4-tiny.weights` (about 23 MB) and `coco.names` into `models/`. They come from the AlexeyAB/darknet releases on GitHub. They are third-party files that were already trained by someone else, and this project does not train anything.

```bash
python main.py download-model
```

### 5. Quick check with the demo video

You do not need a camera for this.

```bash
python main.py generate-demo
python main.py run --source data/demo_synthetic.mp4
```

### 6. Use your own footage

```bash
# a video file
python main.py run --source path/to/your_video.mp4

# a webcam (0 is normally the built-in camera)
python main.py run --source 0

# open a live preview window (needs a desktop environment)
python main.py run --source path/to/video.mp4 --show
```

Whatever you run, the results end up in `output/`:

| File | What it contains |
|---|---|
| `annotated_output.mp4` | The video with boxes, IDs and the counting line drawn on it |
| `detections.csv` | One row per detection: frame, timestamp, object ID, class, confidence, box |
| `summary.json` | Totals for the run: FPS, per-class counts, line crossings |
| `motion_heatmap.jpg` | The accumulated motion heatmap laid over the last frame |
| `edges_sample.jpg` | Canny edge map of the first frame processed |
| `logs/visionflow.log` | The full log of the run |

### A note on the demo video

`generate-demo` draws simple shapes moving over a plain background. That is enough to check that the pipeline runs end to end, but YOLO was trained on real photos, so it will not find any COCO objects in it. To get meaningful detections, use a real clip with people, vehicles or ordinary objects. A phone recording or dashcam footage will do, or you can try Intel's sample file [`person-bicycle-car-detection.mp4`](https://github.com/intel-iot-devkit/sample-videos).

## Running the tests

There are 18 tests. The detector tests use a stand-in for the network, so they run offline and finish quickly without the model download.

```bash
pip install pytest
pytest tests/ -v
```

You should see `18 passed`.

If you also want to see real detections, download the model first (step 4) and then run the pipeline on a video and look inside `output/`:

```bash
python main.py download-model
python main.py run --source data/demo_synthetic.mp4
cat output/summary.json
```

## Non-functional requirements

- **Performance:** the detector is timed with a `@timeit` decorator, average FPS goes into `summary.json`, and the `PROCESS_EVERY_N_FRAMES` setting lets slower machines skip frames to keep up.
- **Reliability:** missing model files, unreadable video sources and frames with no detections are all handled without the program crashing.
- **Usability:** there is one documented entry point, `main.py`, and every subcommand supports `--help`. Nothing needs a GUI to run or be graded.
- **Maintainability:** each module has a single job (`detector`, `tracker`, `analytics`, `report_generator`), the settings live in `config.py`, and the logic has no hard-coded magic numbers.
- **Logging:** timestamped, levelled logs are written to the console and to `output/logs/`.
- **Resource use:** inference runs on the CPU with the small YOLOv4-tiny model rather than full YOLOv4, and frames can be resized to limit the cost per frame.
- **Security:** video never leaves your machine. The only network access is the one-time `download-model` step that fetches public files.

## Why I made these choices

- **YOLOv4-tiny instead of full YOLOv4.** It gives usable speed on a CPU, which is what the grading setup has, and it still covers all 80 COCO classes. For a course project that seemed the sensible trade.
- **A centroid tracker instead of a re-identification model.** It is simple to follow, needs no extra model, and works well enough for one camera watching a scene that is not too crowded. It keeps the project the right size.
- **Classical analytics next to the neural network.** The syllabus covers image enhancement, edge detection and background subtraction with motion analysis (Modules 1, 3 and 4). Using Canny and MOG2 shows those techniques directly instead of leaning only on the detector.

## Screenshots and diagrams

The architecture, workflow, class, use case and sequence diagrams are in `docs/diagrams/`. Sample output frames are in the project report.

## License and credits

This is a coursework submission for CSE3010 Computer Vision (VITyarthi). The YOLOv4-tiny weights come from AlexeyAB/darknet and are under GPL-3.0. Check that repository for the licence terms of the weights.
