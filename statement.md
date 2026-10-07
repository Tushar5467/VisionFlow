# Problem Statement

## Problem Statement

Manually reviewing video footage to count people or vehicles, understand
movement patterns, or spot activity hotspots is slow and error-prone.
Simple motion-triggered alarms (classical background subtraction alone)
cannot distinguish *what* moved — a leaf, a shadow, and a person all trigger
the same alert. Conversely, running a full deep object detector on every
frame with no memory of previous frames wastes computation and cannot answer
"how many distinct people crossed this line?" — it can only say "a person is
present in this frame."

**VisionFlow** addresses this gap: it combines a lightweight, pre-trained
deep object detector (YOLOv4-tiny) with classical multi-object tracking and
motion analytics, producing not just per-frame detections but persistent
object identities, directional counts, and structured, exportable reports —
all from the command line, with no GUI dependency and no model training
required.

## Scope of the Project

**In scope:**
- Detecting 80 COCO object classes (people, vehicles, animals, everyday
  objects) in a video file or webcam stream using a pre-trained YOLOv4-tiny
  model via OpenCV's DNN module.
- Assigning persistent IDs to detected objects across frames using
  centroid-distance tracking.
- Counting directional line crossings (e.g., entries vs. exits).
- Classical CV analytics: Canny edge detection and MOG2-based background
  subtraction with a decaying motion heatmap.
- Exporting per-detection CSV logs and an aggregate JSON summary.
- A CLI for downloading the model, generating a smoke-test video, and
  running the full pipeline.

**Out of scope:**
- Training or fine-tuning any neural network (the project uses inference
  only, on an off-the-shelf model, by design).
- Re-identification across camera views or after long occlusions.
- A graphical user interface (the submission requirements explicitly call
  for CLI executability).
- Cloud deployment, multi-camera fusion, or database persistence beyond
  flat-file CSV/JSON.

## Target Users

- **Course evaluators** assessing the CSE3010 Computer Vision course project
  against the stated rubric (functional/non-functional requirements, design
  documentation, implementation quality, GitHub hygiene, and reporting).
- **Small-scale surveillance / footfall-analysis scenarios** — e.g. a shop
  owner or facility manager who wants a lightweight, self-hosted way to
  count foot traffic or vehicle movement from an existing camera feed
  without sending video to a third-party cloud service.
- **Students / hobbyists** learning how a deep detector, a classical
  tracker, and classical CV analytics fit together in one working pipeline.

## High-Level Features

1. **YOLOv4-tiny object detection** — pre-trained, 80-class COCO detector run
   via OpenCV's DNN module (no training).
2. **Multi-object tracking** — centroid-based tracker assigning stable IDs
   across frames, with automatic deregistration of objects that leave the
   frame.
3. **Line-crossing counter** — directional (in/out) counting of tracked
   objects crossing a configurable horizontal line.
4. **Classical CV analytics** — Canny edge detection and MOG2 background
   subtraction feeding a decaying motion heatmap.
5. **Structured reporting** — CSV per-detection audit trail and a JSON
   run summary (class counts, average FPS, elapsed time, line crossings).
6. **CLI-first design** — `download-model`, `generate-demo`, and `run`
   subcommands; fully executable from a terminal with no GUI dependency.
7. **Automated test suite** — 18 pytest unit tests covering the tracker,
   analytics, report generator, and detector (including a fully mocked,
   offline-runnable detector test path).
