"""End-to-end restoration pipeline.

Stage order is deliberate:

1. prepare   -- deskew, drop the frame, kill the cast, denoise
2. defects   -- inpaint dust and scratches
3. faces     -- generative face restoration at native resolution
4. upscale   -- Real-ESRGAN over the whole frame
5. grade     -- luminance-only contrast and sharpening -> monochrome master
6. colorize  -- DDColor chroma laid over that master

Faces are restored before super resolution because the restorers work on a
fixed 512x512 aligned crop: upscaling first would feed them an already
hallucinated face. Colourisation runs last because it only adds chroma, so it
must sit on top of the final luminance, not be resampled by later stages.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import colorize as colorize_module
from . import defects as defects_module
from . import faces as faces_module
from . import grade as grade_module
from . import prep as prep_module
from . import upscale as upscale_module
from .config import RestoreConfig


class Stopwatch:
    """Records wall time per stage so a run can be profiled from its report."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = round(now - self._start, 2)
        self._start = now


def restore(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Restore one photograph.

    Returns the monochrome master, the colourised version, and a run report.
    When colourisation is disabled both returned frames are the master.
    """
    watch = Stopwatch()
    report: dict = {"config": asdict(config)}

    prepared, prep_report = prep_module.prepare(image, config)
    report["prepare"] = prep_report
    watch.lap("prepare")

    cleaned, defect_report = defects_module.repair(prepared, config)
    report["defects"] = defect_report
    watch.lap("defects")

    restored, face_report = faces_module.restore_faces(cleaned, config)
    report["faces"] = face_report
    watch.lap("faces")

    upscaled, upscale_report = upscale_module.upscale(restored, config)
    report["upscale"] = upscale_report
    watch.lap("upscale")

    master, grade_report = grade_module.grade(upscaled, config)
    report["grade"] = grade_report
    watch.lap("grade")

    if config.colorizer == "none":
        coloured = master
        report["colorize"] = {"model": "none"}
    else:
        coloured, colour_report = colorize_module.colorize(master, config)
        report["colorize"] = colour_report
    watch.lap("colorize")

    report["timings_seconds"] = watch.timings
    report["output_size"] = master.shape[1::-1]
    return master, coloured, report


def comparison_sheet(original: np.ndarray, master: np.ndarray, coloured: np.ndarray,
                     height: int = 1400) -> np.ndarray:
    """Before/after contact sheet at a common height, labelled."""
    def fit(frame: np.ndarray) -> np.ndarray:
        scale = height / frame.shape[0]
        return cv2.resize(
            frame, (round(frame.shape[1] * scale), height), interpolation=cv2.INTER_AREA
        )

    panels = [fit(original), fit(master), fit(coloured)]
    labels = ["ORIGINAL", "RESTORED (B&W)", "RESTORED (COLOUR)"]
    sheet = np.hstack(panels)
    offset = 0
    for panel, label in zip(panels, labels):
        cv2.rectangle(sheet, (offset, 0), (offset + panel.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(
            sheet, label, (offset + 14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
        offset += panel.shape[1]
    return sheet


def save(path: Path, image: np.ndarray, quality: int) -> Path:
    """Write JPEG or PNG based on the suffix; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"failed to write {path}")
    return path
