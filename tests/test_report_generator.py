"""Unit tests for src.report_generator.ReportGenerator."""

import csv
import json
import os

import pytest

from src.report_generator import ReportGenerator


@pytest.fixture
def temp_output_dir(tmp_path):
    return str(tmp_path)


def test_csv_header_written(temp_output_dir):
    reporter = ReportGenerator(output_dir=temp_output_dir)
    reporter.finalize()

    with open(reporter.csv_path, newline="") as f:
        header = next(csv.reader(f))
    assert header == [
        "frame", "timestamp", "object_id", "class_name", "confidence", "x", "y", "w", "h"
    ]


def test_log_frame_writes_rows_and_json_summary(temp_output_dir):
    reporter = ReportGenerator(output_dir=temp_output_dir)
    reporter.log_frame(1, [(0, "person", 0.91, (10, 10, 30, 60))])
    reporter.log_frame(2, [(0, "person", 0.88, (12, 11, 30, 60)),
                            (1, "car", 0.76, (200, 150, 80, 40))])
    summary = reporter.finalize()

    assert os.path.isfile(reporter.csv_path)
    assert os.path.isfile(reporter.json_path)

    with open(reporter.csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 4  # header + 3 detection events

    with open(reporter.json_path) as f:
        json_summary = json.load(f)

    assert json_summary["frames_processed"] == 2
    assert json_summary["total_detection_events"] == 3
    assert json_summary["detections_by_class"]["person"] == 2
    assert json_summary["detections_by_class"]["car"] == 1
    assert summary == json_summary


def test_finalize_includes_extra_stats(temp_output_dir):
    reporter = ReportGenerator(output_dir=temp_output_dir)
    summary = reporter.finalize(extra_stats={"line_crossings_in": 5})
    assert summary["line_crossings_in"] == 5
