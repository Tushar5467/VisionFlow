"""Unit tests for src.analytics (EdgeDetector, MotionAnalyzer)."""

import numpy as np
import pytest

from src.analytics import EdgeDetector, MotionAnalyzer


@pytest.fixture
def sample_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[30:90, 40:120] = (255, 255, 255)  # a bright rectangle on black bg
    return frame


def test_edge_detector_output_shape_and_type(sample_frame):
    edges = EdgeDetector().process(sample_frame)
    assert edges.shape == sample_frame.shape[:2]
    assert edges.dtype == np.uint8


def test_edge_detector_finds_edges_on_high_contrast_shape(sample_frame):
    edges = EdgeDetector().process(sample_frame)
    assert np.count_nonzero(edges) > 0


def test_edge_detector_no_edges_on_blank_frame():
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    edges = EdgeDetector().process(blank)
    assert np.count_nonzero(edges) == 0


def test_motion_analyzer_heatmap_initializes_on_first_frame(sample_frame):
    analyzer = MotionAnalyzer()
    assert analyzer.heatmap is None
    analyzer.process(sample_frame)
    assert analyzer.heatmap is not None
    assert analyzer.heatmap.shape == sample_frame.shape[:2]


def test_motion_analyzer_foreground_mask_shape(sample_frame):
    analyzer = MotionAnalyzer(sample_frame.shape)
    mask = analyzer.process(sample_frame)
    assert mask.shape == sample_frame.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})


def test_motion_analyzer_heatmap_reacts_to_moving_object():
    analyzer = MotionAnalyzer()
    still_bg = np.full((100, 100, 3), 20, dtype=np.uint8)

    for _ in range(15):
        analyzer.process(still_bg.copy())

    moving_frame = still_bg.copy()
    moving_frame[40:60, 40:60] = (255, 255, 255)
    analyzer.process(moving_frame)

    assert analyzer.heatmap[40:60, 40:60].sum() > 0
