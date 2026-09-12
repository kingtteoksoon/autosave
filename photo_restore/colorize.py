"""Learned colourisation for monochrome originals.

Only chroma is taken from the network. Luminance stays exactly as the restored
grey master produced it, so colourisation can never soften detail or shift
exposure -- it adds an a/b layer and nothing else.

DDColor is the one supported network. DeOldify's ONNX export was evaluated and
rejected: its output tensor uses an undocumented range that could not be
denormalised to chroma reliably, and shipping an unverified colour transform is
worse than shipping one model that is known to be correct.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import runtime
from .config import RestoreConfig


def _model_chroma(session, grey: np.ndarray, size: int) -> np.ndarray:
    """Predicted a/b planes at model resolution for a BGR grey frame."""
    scaled = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    normalised = scaled.astype(np.float32) / 255.0
    lightness = cv2.cvtColor(normalised, cv2.COLOR_BGR2LAB)[:, :, :1]
    # DDColor expects the L plane re-rendered as an RGB image, matching the
    # reference implementation's grey-Lab -> RGB round trip.
    grey_lab = np.concatenate((lightness, np.zeros_like(lightness), np.zeros_like(lightness)), axis=-1)
    grey_rgb = cv2.cvtColor(grey_lab, cv2.COLOR_LAB2RGB)

    output = runtime.from_nchw(session.run(None, {"input": runtime.to_nchw(grey_rgb)})[0])
    if output.shape[-1] != 2:
        raise ValueError(
            f"colorizer must emit two chroma planes, got {output.shape[-1]}"
        )
    return output


def _guided_filter(source: np.ndarray, guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-aware smoothing of ``source`` steered by a single-channel ``guide``.

    He et al.'s guided filter, built from box filters so it needs no contrib
    module. Applied to the predicted a/b planes it removes colour speckle and
    the fringes that appear along high-contrast edges, while leaving regional
    colour decisions untouched.
    """
    window = (radius, radius)
    mean_guide = cv2.boxFilter(guide, -1, window)
    mean_source = cv2.boxFilter(source, -1, window)
    corr_guide = cv2.boxFilter(guide * guide, -1, window)
    corr_cross = cv2.boxFilter(guide[:, :, None] * source, -1, window)

    var_guide = corr_guide - mean_guide * mean_guide
    cov = corr_cross - mean_guide[:, :, None] * mean_source
    a = cov / (var_guide + eps)[:, :, None]
    b = mean_source - a * mean_guide[:, :, None]
    return cv2.boxFilter(a, -1, window) * guide[:, :, None] + cv2.boxFilter(b, -1, window)


def _chroma_bias(chroma: np.ndarray, lightness: np.ndarray, quantile: float = 0.97) -> np.ndarray:
    """Estimate a global colour cast from the frame's brightest pixels.

    The white-patch assumption: in a photograph the brightest surfaces are
    normally close to neutral, so any chroma they carry is the network's own
    tint rather than the scene's. Colourisers trained on modern photographs
    reliably drift warm or magenta on studio portraits, and this measures that
    drift where it is least confusable with real colour.

    The assumption fails whenever the brightest surface is a large coloured
    one -- a lit studio backdrop rather than a white collar -- and it fails
    loudly, returning a big shift that would drag skin far off natural. That is
    why the caller caps the correction hard: a small measured tint is worth
    removing, a large one is evidence the assumption does not hold here.
    """
    threshold = np.quantile(lightness, quantile)
    highlights = chroma[lightness >= threshold]
    if highlights.size < 64:
        return np.zeros(2, dtype=np.float32)
    return np.median(highlights.reshape(-1, 2), axis=0).astype(np.float32)


def _shadow_rolloff(lightness: np.ndarray, floor: float, knee: float) -> np.ndarray:
    """Per-pixel chroma scale that fades colour out of the deep shadows.

    Dark areas hold the least reliable colour evidence, and a coloriser's worst
    guesses land there: an unlit studio backdrop comes back violet far more
    often than it comes back grey. Real photographic shadows also desaturate
    toward black, so damping chroma by lightness corrects the artefact and
    matches how film behaves.
    """
    if floor >= 1.0:
        return np.ones_like(lightness)
    base = 0.08
    weight = np.clip((lightness - base) / max(knee - base, 1e-3), 0.0, 1.0)
    return floor + (1.0 - floor) * weight


def colorize(grey: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, dict]:
    """Colourise a monochrome BGR frame, preserving its luminance exactly."""
    report: dict = {"model": config.colorizer}
    if config.colorizer == "none":
        return grey, report

    session = runtime.load(config.colorizer, config.intra_op_threads)
    chroma = _model_chroma(session, grey, config.colorizer_size)
    height, width = grey.shape[:2]
    chroma = cv2.resize(chroma, (width, height), interpolation=cv2.INTER_CUBIC)

    lightness = cv2.cvtColor(grey.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, :1]
    radius = int(width * config.chroma_refine_radius)
    if radius >= 2:
        chroma = _guided_filter(chroma, lightness[:, :, 0] / 100.0, radius, 1e-4)
        report["chroma_refine_radius_px"] = radius
    if config.white_balance:
        bias = _chroma_bias(chroma, lightness[:, :, 0])
        magnitude = float(np.linalg.norm(bias))
        if magnitude > config.white_balance_limit:
            bias = bias * (config.white_balance_limit / magnitude)
        chroma = chroma - bias * float(config.white_balance_strength)
        report["white_balance_shift"] = [round(float(v), 2) for v in bias]

    rolloff = _shadow_rolloff(
        lightness[:, :, 0] / 100.0, config.shadow_chroma_floor, config.shadow_chroma_knee
    )
    chroma = chroma * rolloff[:, :, None] * float(config.chroma_strength)

    lab = np.concatenate((lightness, chroma), axis=-1).astype(np.float32)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    report["chroma_strength"] = config.chroma_strength
    report["chroma_abs_mean"] = round(float(np.abs(chroma).mean()), 2)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8), report
