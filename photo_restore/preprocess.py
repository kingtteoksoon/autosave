"""Physical clean-up of a photographed or scanned print.

These steps run before any neural model: they undo capture artefacts (frame,
lighting gradient, colour cast of the ambient light, grain) so the restoration
networks see a clean, neutral input instead of amplifying the damage.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_bgr(path: str) -> np.ndarray:
    """Read an image honouring the EXIF orientation flag."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def crop_frame(bgr: np.ndarray, coverage: float = 0.04, pad: int = 6) -> tuple[np.ndarray, dict]:
    """Trim gilt/ornate frame borders captured around the print.

    A frame band is detected as a border row/column where a large share of
    pixels are saturated warm (gold/wood) tones, which the monochrome print
    itself never contains.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16))
    frame_like = ((hue >= 8) & (hue <= 45) & (sat > 60) & (val > 55)).astype(np.float32)
    height, width = frame_like.shape

    def band(profile: np.ndarray, limit: int) -> int:
        hits = np.nonzero(profile[:limit] > coverage)[0]
        return int(hits[-1]) + 1 if hits.size else 0

    rows, cols = frame_like.mean(axis=1), frame_like.mean(axis=0)
    box = dict(top=band(rows, int(height * 0.12)), bottom=band(rows[::-1], int(height * 0.12)),
               left=band(cols, int(width * 0.12)), right=band(cols[::-1], int(width * 0.12)))

    def clip(value: int, extent: int) -> int:
        return min(value + pad, extent // 4) if value else 0

    top, bottom = clip(box["top"], height), clip(box["bottom"], height)
    left, right = clip(box["left"], width), clip(box["right"], width)
    return np.ascontiguousarray(bgr[top : height - bottom, left : width - right]), box


def trim_dark_edges(image: np.ndarray, threshold_ratio: float = 0.2,
                    max_fraction: float = 0.02) -> tuple[np.ndarray, dict]:
    """Drop the near-black sliver a frame's inner lip leaves along a border.

    Run *after* tone normalisation, where that sliver collapses to near zero
    while genuinely dark picture content (a black suit, a vignetted backdrop)
    stays well above it. Contiguous from the edge and capped at
    ``max_fraction`` per side, so it can never bite into the photograph.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    height, width = gray.shape

    def scan(profile: np.ndarray, limit: int) -> int:
        interior = float(np.median(profile[int(len(profile) * 0.2) : int(len(profile) * 0.8)]))
        cutoff = threshold_ratio * interior
        band = 0
        for index in range(limit):
            if profile[index] >= cutoff:
                break
            band = index + 1
        return band

    rows, cols = gray.mean(axis=1), gray.mean(axis=0)
    edges = dict(top=scan(rows, int(height * max_fraction)), bottom=scan(rows[::-1], int(height * max_fraction)),
                 left=scan(cols, int(width * max_fraction)), right=scan(cols[::-1], int(width * max_fraction)))
    trimmed = image[edges["top"] : height - edges["bottom"], edges["left"] : width - edges["right"]]
    return np.ascontiguousarray(trimmed), edges


MONOCHROME_THRESHOLD = 15.0


def monochrome_score(bgr: np.ndarray) -> float:
    """0 = monochrome print, higher = genuinely coloured original.

    The global a/b median is removed first, so a uniform colour cast - ambient
    light, sepia toning, a yellowed print - does not count as colour. The 75th
    percentile of what remains is used rather than a maximum, so that a few
    saturated outliers (a frame reflection, chroma noise) cannot flip a
    black-and-white print into the colour branch. Measured values: monochrome
    print ~5, colour photograph ~45+.
    """
    scale = 512.0 / max(bgr.shape[:2])
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    a, b = a - np.median(a), b - np.median(b)
    return float(np.percentile(np.sqrt(a * a + b * b), 75))


def to_luma(bgr: np.ndarray) -> np.ndarray:
    """Perceptual luminance; discards the capture's colour cast for a mono print."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]


def flat_field(gray: np.ndarray, sigma_ratio: float = 0.12) -> np.ndarray:
    """Divide out the low-frequency lighting gradient of the capture.

    The Gaussian is deliberately wide rather than infinite: a fully flat field
    would also erase the photographer's intended lighting falloff, so roughly
    80% of a linear gradient is removed and the modelled studio lighting stays.
    """
    values = gray.astype(np.float32)
    sigma = max(gray.shape) * sigma_ratio
    illumination = cv2.GaussianBlur(values, (0, 0), sigma)
    corrected = values / np.maximum(illumination, 1e-3) * float(illumination.mean())
    return np.clip(corrected, 0, 255)


def denoise(gray: np.ndarray, strength: int = 5) -> np.ndarray:
    """Non-local-means: removes sensor/JPEG grain while keeping print detail."""
    return cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, h=strength, templateWindowSize=7, searchWindowSize=21)


def normalize_tone(gray: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5, clahe_clip: float = 1.2,
                   clahe_mix: float = 0.45) -> np.ndarray:
    """Re-set black/white points, then add a gentle local-contrast recovery."""
    values = gray.astype(np.float32)
    lo, hi = np.percentile(values, lo_pct), np.percentile(values, hi_pct)
    stretched = np.clip((values - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255).astype(np.uint8)
    if clahe_mix <= 0:
        return stretched
    local = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(stretched)
    return cv2.addWeighted(stretched, 1.0 - clahe_mix, local, clahe_mix, 0)


def unsharp(gray: np.ndarray, amount: float = 0.45, sigma: float = 1.6) -> np.ndarray:
    """Mild detail recovery; kept low because the SR pass adds the real detail."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    sharpened = gray.astype(np.float32) * (1 + amount) - blurred * amount
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def prepare(bgr: np.ndarray, *, do_crop: bool = True, denoise_strength: int = 5) -> tuple[np.ndarray, dict]:
    """Full pre-processing chain -> 3-channel neutral image ready for the models."""
    report: dict = {}
    working = bgr
    if do_crop:
        working, report["frame_band"] = crop_frame(working)
    report["mono_score"] = monochrome_score(working)
    report["is_monochrome"] = report["mono_score"] < MONOCHROME_THRESHOLD
    if report["is_monochrome"]:
        luma = to_luma(working)
        luma = flat_field(luma)
        luma = denoise(luma, denoise_strength)
        luma = normalize_tone(luma)
        prepared = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
    else:
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        lab[..., 0] = normalize_tone(denoise(flat_field(lab[..., 0])))
        prepared = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    prepared, report["dark_edges"] = trim_dark_edges(prepared)
    report["size"] = prepared.shape[1::-1]
    return prepared, report
