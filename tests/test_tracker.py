"""Unit tests for src.tracker (CentroidTracker, LineCrossingCounter)."""

from dataclasses import dataclass

import pytest

from src.tracker import CentroidTracker, LineCrossingCounter


@dataclass
class FakeDetection:
    box: tuple


def test_register_new_objects_on_first_frame():
    tracker = CentroidTracker()
    detections = [FakeDetection(box=(10, 10, 20, 20)), FakeDetection(box=(100, 100, 20, 20))]
    boxes = tracker.update(detections)
    assert len(boxes) == 2
    assert set(boxes.values()) == {(10, 10, 20, 20), (100, 100, 20, 20)}


def test_same_object_keeps_same_id_across_frames():
    tracker = CentroidTracker()
    tracker.update([FakeDetection(box=(10, 10, 20, 20))])
    first_id = next(iter(tracker.objects.keys()))

    # Object moves slightly -> should be matched to the same ID.
    tracker.update([FakeDetection(box=(14, 12, 20, 20))])
    second_id = next(iter(tracker.objects.keys()))

    assert first_id == second_id
    assert len(tracker.objects) == 1


def test_object_deregistered_after_max_disappeared():
    tracker = CentroidTracker(max_disappeared=2, max_distance=50)
    tracker.update([FakeDetection(box=(10, 10, 20, 20))])
    assert len(tracker.objects) == 1

    # No detections for several consecutive frames.
    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert len(tracker.objects) == 0


def test_far_away_detection_gets_new_id():
    tracker = CentroidTracker(max_distance=30)
    tracker.update([FakeDetection(box=(10, 10, 20, 20))])
    tracker.update([FakeDetection(box=(500, 500, 20, 20))])
    assert len(tracker.objects) == 2


def test_line_crossing_counter_detects_downward_crossing():
    counter = LineCrossingCounter(line_y=100)
    # Object moves from above the line to below it across three frames.
    counter.update({1: (50, 80, 20, 20)})
    counter.update({1: (50, 95, 20, 20)})
    counter.update({1: (50, 120, 20, 20)})

    assert counter.count_down == 1
    assert counter.count_up == 0
    assert counter.total == 1


def test_line_crossing_counter_only_counts_once_per_object():
    counter = LineCrossingCounter(line_y=100)
    counter.update({1: (50, 80, 20, 20)})
    counter.update({1: (50, 120, 20, 20)})
    counter.update({1: (50, 130, 20, 20)})  # still below the line, no re-count
    assert counter.count_down == 1
