"""Dust and scratch removal.

Prints accumulate specks, and a phone capture adds its own. These read as small
high-contrast blobs sitting on parts of the picture that are otherwise smooth,
which is exactly what makes them separable from real detail: an eye catchlight
or a tie pattern lives in busy neighbourhoods, a dust mote on a studio backdrop
does not.

Smoothness is measured on the median-filtered frame rather than the original.
Measured on the original, every speck would inflate the local variance around
itself and so exclude itself from detection.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import RestoreConfig


def _evidence(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel deviation from the local median, and where the picture is smooth.

    Both are independent of the detection threshold, so they are computed once
    and reused while the threshold is tuned.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    median = cv2.medianBlur(grey, 5).astype(np.float32)
    residual = np.abs(grey.astype(np.float32) - median)

    detail = median - cv2.GaussianBlur(median, (0, 0), 4.0)
    local_std = np.sqrt(cv2.GaussianBlur(detail * detail, (0, 0), 8.0))
    # ``<=`` with a floor matters: on a perfectly flat region every local
    # standard deviation is zero, and a strict ``<`` comparison against a zero
    # percentile would mark nothing as smooth and so find no defects at all.
    limit = max(float(np.percentile(local_std, config.defect_smooth_percentile)), 1e-6)
    return residual, local_std <= limit


def _mask_from_evidence(
    residual: np.ndarray, smooth: np.ndarray, config: RestoreConfig, threshold: float
) -> np.ndarray:
    """Speck mask at one threshold, filtered by blob size and extent."""
    candidates = ((residual > threshold) & smooth).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)

    height, width = residual.shape[:2]
    # The floor keeps the size gate meaningful on small images, where the
    # area fraction alone would reject specks of any realistic size.
    max_area = max(25, int(height * width * config.defect_max_area_fraction))
    max_extent = max(12, int(min(height, width) * 0.02))

    keep = [
        index
        for index in range(1, count)
        if 2 <= stats[index, cv2.CC_STAT_AREA] <= max_area
        and max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]) <= max_extent
    ]
    mask = np.zeros_like(candidates)
    if keep:
        mask[np.isin(labels, keep)] = 1
    return mask


def detect(
    image: np.ndarray, config: RestoreConfig, threshold: float | None = None
) -> np.ndarray:
    """Binary mask of speck-like defects."""
    residual, smooth = _evidence(image, config)
    level = config.defect_threshold if threshold is None else threshold
    return _mask_from_evidence(residual, smooth, config, level)


def repair(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, dict]:
    """Detect and inpaint specks; returns the repaired frame and a report."""
    report: dict = {"enabled": config.remove_defects}
    if not config.remove_defects:
        return image, report

    # Dust is rare. A detection covering a large share of the frame means the
    # threshold has caught the print's own grain, so raise it until what is
    # left is plausibly damage. Inpainting grain would smooth away real texture
    # across the whole picture, which is the denoiser's job to temper, not
    # this stage's job to erase.
    residual, smooth = _evidence(image, config)
    threshold = config.defect_threshold
    for _ in range(5):
        mask = _mask_from_evidence(residual, smooth, config, threshold)
        if mask.mean() <= config.defect_max_frame_fraction:
            break
        threshold *= 1.6

    pixels = int(mask.sum())
    report["defect_threshold"] = round(threshold, 2)
    report["defect_pixels"] = pixels
    report["defect_fraction"] = round(pixels / mask.size, 6)
    if pixels == 0:
        return image, report

    # Dilate so the inpainting reaches past each speck's soft edge.
    grown = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image, grown, 3, cv2.INPAINT_TELEA), report
