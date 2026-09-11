"""Model-free unit tests for the deterministic parts of the pipeline.

Run with:  python3 -m pytest tests -q   (or simply: python3 tests/test_units.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photo_restore import preprocess
from photo_restore.faces import FFHQ_TEMPLATE, _iou
from photo_restore.upscale import _ramp


def _framed_photo() -> np.ndarray:
    """Grey portrait-shaped print surrounded by a saturated gold frame."""
    image = np.full((400, 300, 3), 128, dtype=np.uint8)
    image[:20] = image[-20:] = (30, 170, 220)      # BGR gold
    image[:, :15] = image[:, -15:] = (30, 170, 220)
    return image


def test_crop_frame_removes_gold_border() -> None:
    cropped, bands = preprocess.crop_frame(_framed_photo())
    assert bands["top"] >= 20 and bands["bottom"] >= 20
    assert bands["left"] >= 15 and bands["right"] >= 15
    assert cropped.shape[0] < 400 and cropped.shape[1] < 300
    assert preprocess.monochrome_score(cropped) < 5.0


def test_monochrome_detection() -> None:
    gray = cv2.cvtColor(np.random.randint(0, 255, (64, 64), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    sepia = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)
    sepia[..., 1], sepia[..., 2] = 140, 160                 # uniform warm cast
    sepia = cv2.cvtColor(sepia, cv2.COLOR_LAB2BGR)
    bars = np.zeros((64, 64, 3), np.uint8)                  # real colour content
    for index, colour in enumerate([(220, 30, 30), (30, 220, 30), (30, 30, 220), (30, 200, 220)]):
        bars[:, index * 16 : (index + 1) * 16] = colour

    assert preprocess.monochrome_score(gray) < 5.0
    assert preprocess.monochrome_score(sepia) < preprocess.MONOCHROME_THRESHOLD   # a cast is not colour
    assert preprocess.monochrome_score(bars) > preprocess.MONOCHROME_THRESHOLD


def test_flat_field_removes_lighting_gradient() -> None:
    gradient = np.tile(np.linspace(40, 210, 256, dtype=np.float32), (256, 1))
    corrected = preprocess.flat_field(gradient.astype(np.uint8))
    before = gradient[:, -40:].mean() - gradient[:, :40].mean()
    after = corrected[:, -40:].mean() - corrected[:, :40].mean()
    assert after < 0.3 * before                              # gradient largely removed
    assert after > 0                                         # but lighting direction preserved


def test_normalize_tone_expands_range() -> None:
    flat = np.random.randint(90, 140, (128, 128)).astype(np.uint8)
    out = preprocess.normalize_tone(flat)
    assert out.min() < 20 and out.max() > 235


def test_ffhq_template_geometry() -> None:
    left_eye, right_eye, nose, left_mouth, right_mouth = FFHQ_TEMPLATE
    assert left_eye[0] < nose[0] < right_eye[0]          # eyes straddle the nose
    assert left_eye[1] < nose[1] < left_mouth[1]         # eyes above nose above mouth
    assert abs(left_mouth[1] - right_mouth[1]) < 2.0     # mouth corners level
    assert FFHQ_TEMPLATE.max() < 512


def test_iou() -> None:
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    assert _iou(box, box) == 1.0
    assert _iou(box, np.array([20, 20, 30, 30], dtype=np.float32)) == 0.0


def test_tile_ramp_partitions_unity() -> None:
    """Overlapping ramps must sum to a constant, otherwise tiles show as banding."""
    tile, feather, step = 64, 16, 48
    accumulator = np.zeros((64, 256, 1), dtype=np.float32)
    for x in range(0, 256 - tile + 1, step):
        accumulator[:, x : x + tile] += _ramp(64, tile, feather)
    interior = accumulator[24:40, tile : 256 - tile]        # rows away from the vertical fade
    assert abs(interior.mean() - 1.0) < 1e-3
    assert interior.std() < 1e-3


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    raise SystemExit(1 if failures else 0)
