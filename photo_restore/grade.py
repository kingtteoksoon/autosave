"""Final tone and detail grading.

Every operation is confined to luminance. Contrast and sharpening applied to
BGR channels independently shift hue and amplify chroma noise, which is the
failure mode that makes a restored photograph look artificial.

Grading is global by default. CLAHE is available but off: on a studio portrait
the background is one large smooth vignette, and equalising it tile by tile
lifts it to mid grey and destroys the depth the photographer lit for.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import RestoreConfig


def set_levels(
    lightness: np.ndarray,
    black_point: float,
    white_point: float,
    anchors: tuple[float, float] = (0.3, 99.7),
) -> np.ndarray:
    """Map the anchor percentiles of L onto the target black and white points."""
    low, high = np.percentile(lightness, anchors)
    if high - low < 1e-3:
        return lightness
    scaled = (lightness.astype(np.float32) - low) / (high - low)
    scaled = scaled * (white_point - black_point) + black_point
    return np.clip(scaled, 0, 255).astype(np.uint8)


def s_curve(lightness: np.ndarray, strength: float) -> np.ndarray:
    """Blend L towards a raised-cosine curve: deepens shadows, holds highlights."""
    if strength <= 0:
        return lightness
    ramp = np.arange(256, dtype=np.float32) / 255.0
    curved = 0.5 * (1.0 - np.cos(np.pi * ramp))
    table = np.clip((1.0 - strength) * ramp + strength * curved, 0.0, 1.0) * 255.0
    return cv2.LUT(lightness, table.astype(np.uint8))


def local_contrast(lightness: np.ndarray, clip: float) -> np.ndarray:
    """CLAHE on the L plane, for flat scans that need shadow detail recovered."""
    if clip <= 0:
        return lightness
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
    return clahe.apply(lightness)


def unsharp(lightness: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Unsharp mask on the L plane."""
    if amount <= 0:
        return lightness
    blurred = cv2.GaussianBlur(lightness.astype(np.float32), (0, 0), float(radius))
    sharpened = lightness.astype(np.float32) * (1.0 + amount) - blurred * amount
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def grade(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, dict]:
    """Set levels, add global contrast, then sharpen; chroma passes through."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, chroma = lab[:, :, 0], lab[:, :, 1:]

    lightness = set_levels(lightness, config.black_point, config.white_point)
    lightness = s_curve(lightness, config.tone_contrast)
    lightness = local_contrast(lightness, config.clahe_clip)
    lightness = unsharp(lightness, config.unsharp_amount, config.unsharp_radius)

    merged = np.concatenate((lightness[:, :, None], chroma), axis=-1)
    report = {
        "levels": [config.black_point, config.white_point],
        "tone_contrast": config.tone_contrast,
        "clahe_clip": config.clahe_clip,
        "unsharp_amount": config.unsharp_amount,
    }
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), report
