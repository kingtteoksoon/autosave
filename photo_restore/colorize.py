"""Colourisation of a monochrome print with DDColor (ONNX, CPU).

The network only predicts chrominance: the restored luminance of the print is
kept and only the a/b channels of CIELAB are replaced, so the tonality of the
original photograph survives.

DDColor has a known global chroma bias on studio portraits (it washes the whole
frame magenta/violet). ``neutralise`` pulls the *median* chroma back towards
neutral, which fixes the cast on the large near-neutral areas (backdrop, suit)
while leaving genuinely coloured minorities such as skin and tie intact.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models import session

MODEL = "ddcolor.onnx"
MODEL_SIZE = 512


def _prepare(bgr: np.ndarray, size: int = MODEL_SIZE) -> np.ndarray:
    """DDColor expects the luminance-only image as RGB in [0, 1]."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    luminance_only = np.concatenate([lab[..., :1], np.zeros_like(lab[..., :2])], axis=-1)
    gray_rgb = np.clip(cv2.cvtColor(luminance_only, cv2.COLOR_LAB2RGB), 0, 1)
    resized = cv2.resize(gray_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)


def predict_chroma(bgr: np.ndarray) -> np.ndarray:
    """Predicted CIELAB a/b planes at the resolution of ``bgr``."""
    chroma = session(MODEL).run(None, {"input": _prepare(bgr)})[0][0].transpose(1, 2, 0)
    return cv2.resize(chroma, bgr.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def neutralized_chroma(bgr: np.ndarray, neutralize: float = 0.7, flip_tta: bool = True) -> np.ndarray:
    """Predicted a/b with DDColor's global cast pulled back towards neutral.

    ``flip_tta`` averages the prediction with the mirrored one. DDColor's bias is
    not left-right symmetric (it tinted one shoulder magenta on this portrait),
    so the average of both orientations is measurably steadier - at the cost of
    one extra forward pass.
    """
    chroma = predict_chroma(bgr)
    if flip_tta:
        chroma = (chroma + cv2.flip(predict_chroma(cv2.flip(bgr, 1)), 1)) / 2.0
    if neutralize:
        bias = np.array([np.median(chroma[..., 0]), np.median(chroma[..., 1])], dtype=np.float32)
        chroma = chroma - neutralize * bias
    return chroma


def apply_chroma(bgr: np.ndarray, chroma: np.ndarray, saturation: float = 0.85) -> np.ndarray:
    """Combine an existing a/b prediction with the luminance of ``bgr``.

    Chrominance is low frequency, so a prediction made at the pre-upscale size
    can be resampled onto a super-resolved luminance without visible loss - and
    it is more reliable there, because super resolution shifts the luminance
    statistics the colour network was calibrated on.
    """
    if chroma.shape[:2] != bgr.shape[:2]:
        chroma = cv2.resize(chroma, bgr.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab[..., 1:] = chroma * saturation
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255.0, 0, 255).astype(np.uint8)


def colorize(bgr: np.ndarray, saturation: float = 0.85, neutralize: float = 0.7,
             flip_tta: bool = True) -> np.ndarray:
    """Return ``bgr`` colourised, keeping its luminance untouched."""
    return apply_chroma(bgr, neutralized_chroma(bgr, neutralize, flip_tta), saturation)
