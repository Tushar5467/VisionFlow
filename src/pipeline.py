"""
pipeline.py
-----------
Wires detector + tracker + analytics + report_generator together into a
single runnable pipeline over a video file or webcam stream.
"""

import os
import time

import cv2

from src import config, utils
from src.detector import YOLODetector
from src.tracker import CentroidTracker, LineCrossingCounter
from src.analytics import EdgeDetector, MotionAnalyzer
from src.report_generator import ReportGenerator

logger = utils.get_logger(__name__)


class VisionFlowPipeline:
    def __init__(self, source, save_video: bool = True, show_window: bool = False):
        self.source = source
        self.save_video = save_video
        self.show_window = show_window

        self.detector = YOLODetector()
        self.tracker = CentroidTracker()
        self.edge_detector = EdgeDetector()
        self.motion_analyzer = None  # needs frame size, init after first frame
        self.reporter = ReportGenerator()
        self.line_counter = None

        self.video_writer = None
        self.heatmap_writer_path = os.path.join(config.OUTPUT_DIR, "motion_heatmap.jpg")
        self.annotated_video_path = os.path.join(config.OUTPUT_DIR, "annotated_output.mp4")
        self.edges_sample_path = os.path.join(config.OUTPUT_DIR, "edges_sample.jpg")

    def _init_video_io(self, cap, frame):
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.motion_analyzer = MotionAnalyzer(frame.shape)
        self.line_counter = LineCrossingCounter(line_y=int(h * config.COUNT_LINE_POSITION))

        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.annotated_video_path, fourcc, fps, (w, h)
            )

    def run(self) -> dict:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        frame_index = 0
        edge_sample_saved = False
        last_good_frame = None
        t_start = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = utils.resize_frame(frame, config.FRAME_RESIZE_WIDTH)
                last_good_frame = frame
                frame_index += 1

                if frame_index == 1:
                    self._init_video_io(cap, frame)

                if frame_index % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                detections = self.detector.detect(frame)
                tracked_boxes = self.tracker.update(detections)
                self.line_counter.update(tracked_boxes)

                # Match tracked boxes back to class/confidence for reporting.
                # (Simple IoU-free match by identical box coordinates, since
                # the tracker stores the box it was just updated with.)
                box_to_detection = {d.box: d for d in detections}
                report_rows = []
                for object_id, box in tracked_boxes.items():
                    det = box_to_detection.get(box)
                    if det is None:
                        continue
                    report_rows.append((object_id, det.class_name, det.confidence, box))
                    label = f"ID {object_id} {det.class_name} {det.confidence:.2f}"
                    utils.draw_box(frame, box, label)

                self.reporter.log_frame(frame_index, report_rows)

                fg_mask = self.motion_analyzer.process(frame)
                edges = self.edge_detector.process(frame)

                if not edge_sample_saved:
                    cv2.imwrite(self.edges_sample_path, edges)
                    edge_sample_saved = True

                cv2.line(
                    frame,
                    (0, self.line_counter.line_y),
                    (frame.shape[1], self.line_counter.line_y),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"In: {self.line_counter.count_down}  Out: {self.line_counter.count_up}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if self.video_writer is not None:
                    self.video_writer.write(frame)

                if self.show_window:
                    cv2.imshow("VisionFlow", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_index % 25 == 0:
                    logger.info(f"Processed frame {frame_index}")

        finally:
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            if self.show_window:
                cv2.destroyAllWindows()

            if (
                self.motion_analyzer is not None
                and self.motion_analyzer.heatmap is not None
                and last_good_frame is not None
            ):
                # Use last successfully read frame as background for the heatmap overlay.
                cv2.imwrite(
                    self.heatmap_writer_path,
                    self.motion_analyzer.get_heatmap_overlay(last_good_frame),
                )

        elapsed = round(time.perf_counter() - t_start, 2)
        summary = self.reporter.finalize(
            extra_stats={
                "line_crossings_in": self.line_counter.count_down if self.line_counter else 0,
                "line_crossings_out": self.line_counter.count_up if self.line_counter else 0,
                "processing_time_seconds": elapsed,
            }
        )
        logger.info(f"Pipeline finished in {elapsed}s over {frame_index} frames.")
        return summary
