"""Smoke test for the line segmentation pipeline.

Generates a synthetic page image with a few horizontal text-like rectangles
(simulating lines of handwriting) and verifies that segmentation returns at
least one line crop.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from handwriting.segmentation import segment_lines


def make_synthetic_page(
    n_lines: int = 4,
    page_h: int = 800,
    page_w: int = 600,
) -> np.ndarray:
    """Create a white page with *n_lines* dark rectangles simulating text lines."""
    img = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    line_height = 30
    gap = (page_h - n_lines * line_height) // (n_lines + 1)
    for i in range(n_lines):
        y = gap + i * (line_height + gap)
        x0, x1 = 50, page_w - 50
        cv2.rectangle(img, (x0, y), (x1, y + line_height), (20, 20, 20), thickness=-1)
    return img


def test_segment_returns_lines(tmp_path: Path):
    page = make_synthetic_page(n_lines=4)
    lines = segment_lines(page, tmp_path, page_id="smoke_test")
    assert len(lines) >= 1, "Expected at least 1 line from segmentation"


def test_segment_creates_files(tmp_path: Path):
    page = make_synthetic_page(n_lines=3)
    lines = segment_lines(page, tmp_path, page_id="file_test")
    # Each line crop PNG should exist
    for lb in lines:
        assert Path(lb.image_path).exists(), f"Missing crop: {lb.image_path}"
    # meta.json should be written
    assert (tmp_path / "meta.json").exists()


def test_segment_meta_content(tmp_path: Path):
    page = make_synthetic_page(n_lines=3)
    lines = segment_lines(page, tmp_path, page_id="meta_test")
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["page_id"] == "meta_test"
    assert meta["num_lines"] == len(lines)
    assert len(meta["lines"]) == len(lines)
