"""Source preparation: deskew, frame removal, colour-cast neutralisation, denoise.

A phone snapshot of a framed print carries four defects that the neural stages
downstream all handle badly: the picture frame itself, the tilt of the capture,
a global colour cast from the ambient light, and print/sensor grain. Correcting
them first is what makes face restoration and colourisation behave.

Frame detection works on CIELAB chroma distance from the picture's own interior
chroma. A gilt frame differs from a monochrome print in hue no matter how dim
it is, so the test stays valid across dark vignettes where saturation-based
tests produce false positives.
"""
from __future__ import annotations

import itertools
import random

import cv2
import numpy as np

from .config import RestoreConfig

#: Chroma distances are bucketed over this range before Otsu thresholding.
_CHROMA_SCALE = 64.0
#: A side counts as framed only if its outermost line is almost entirely
#: off-chroma; a real frame spans the full edge, a bright object does not.
_EDGE_PRESENCE = 0.70
#: Within a framed side, lines above this coverage are still frame.
_EDGE_CONTINUATION = 0.50


def chroma_distance(image: np.ndarray) -> np.ndarray:
    """Per-pixel CIELAB chroma distance from the picture interior's median hue."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = lab[:, :, 1:] - 128.0
    height, width = image.shape[:2]
    interior = chroma[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    reference = np.median(interior.reshape(-1, 2), axis=0)
    return np.linalg.norm(chroma - reference, axis=-1)


def _edge_bands(distance: np.ndarray, max_fraction: float) -> dict[str, np.ndarray]:
    """Edge strips, each oriented so that row 0 is the outermost line."""
    height, width = distance.shape
    row_limit = max(1, int(height * max_fraction))
    col_limit = max(1, int(width * max_fraction))
    return {
        "top": distance[:row_limit],
        "bottom": distance[::-1][:row_limit],
        "left": distance[:, :col_limit].T,
        "right": distance[:, ::-1][:, :col_limit].T,
    }


def _otsu_cutoff(bands: dict[str, np.ndarray]) -> float:
    """One self-calibrating cutoff separating frame-coloured from picture pixels."""
    pooled = np.concatenate([band.ravel() for band in bands.values()])
    quantised = (np.clip(pooled / _CHROMA_SCALE, 0.0, 1.0) * 255).astype(np.uint8)
    level, _ = cv2.threshold(quantised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level) / 255.0 * _CHROMA_SCALE


def detect_frame_border(
    image: np.ndarray, max_fraction: float = 0.10
) -> tuple[int, int, int, int]:
    """Pixels of picture frame to trim from each edge, as (top, bottom, left, right)."""
    distance = chroma_distance(image)
    bands = _edge_bands(distance, max_fraction)
    cutoff = _otsu_cutoff(bands)

    borders: dict[str, int] = {}
    for side, band in bands.items():
        coverage = (band > cutoff).mean(axis=1)
        if coverage.size == 0 or coverage[0] < _EDGE_PRESENCE:
            borders[side] = 0
            continue
        last = 0
        for index, value in enumerate(coverage):
            if value >= _EDGE_CONTINUATION:
                last = index + 1
        # The frame's soft shadow edge extends a little past the colour step.
        borders[side] = min(last + 6, band.shape[0])
    return borders["top"], borders["bottom"], borders["left"], borders["right"]


def _robust_line(xs: np.ndarray, ys: np.ndarray, samples: int = 4000) -> tuple[float, float]:
    """Theil-Sen fit: median pairwise slope, then median intercept.

    Chosen over least squares because the frame edge samples contain outliers
    wherever ornament or glare defeats the per-column boundary search.
    """
    pairs = list(itertools.combinations(range(len(xs)), 2))
    if len(pairs) > samples:
        pairs = random.Random(0).sample(pairs, samples)
    slopes = [
        (ys[b] - ys[a]) / (xs[b] - xs[a]) for a, b in pairs if xs[b] != xs[a]
    ]
    slope = float(np.median(slopes)) if slopes else 0.0
    return slope, float(np.median(ys - slope * xs))


def estimate_tilt(image: np.ndarray, max_fraction: float = 0.10) -> float:
    """Capture tilt in degrees, measured from the top frame edge.

    Returns 0.0 when no usable edge is found or the fit is not straight enough
    to trust, so a photograph without a frame is never rotated on noise.
    """
    distance = chroma_distance(image)
    bands = _edge_bands(distance, max_fraction)
    cutoff = _otsu_cutoff(bands)
    band = cv2.GaussianBlur(bands["top"], (0, 0), 2.0)
    if (band > cutoff).mean(axis=1)[0] < _EDGE_PRESENCE:
        return 0.0

    xs_list: list[float] = []
    ys_list: list[float] = []
    for x in range(0, band.shape[1], 5):
        hits = np.where(band[:, x] > cutoff)[0]
        if hits.size:
            xs_list.append(float(x))
            ys_list.append(float(hits.max()))
    if len(xs_list) < 20:
        return 0.0

    xs = np.asarray(xs_list)
    ys = np.asarray(ys_list)
    slope, intercept = _robust_line(xs, ys)
    residual = np.abs(ys - (slope * xs + intercept))
    # A straight frame rail fits tightly; a ragged profile means we measured
    # ornament rather than the boundary, so decline to rotate.
    if float(np.percentile(residual, 75)) > 0.05 * band.shape[0] + 5.0:
        return 0.0
    angle = float(np.degrees(np.arctan(slope)))
    return angle if abs(angle) <= 5.0 else 0.0


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by ``-angle`` and trim to the largest axis-aligned inscribed rectangle.

    Trimming avoids the replicated-edge smear that rotation otherwise leaves in
    the corners, at the cost of a few per cent of the frame.
    """
    if abs(angle) < 0.05:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    radians = abs(np.radians(angle))
    sin, cos = np.sin(radians), np.cos(radians)
    long_side, short_side = max(width, height), min(width, height)
    if short_side <= 2 * sin * cos * long_side or abs(sin - cos) < 1e-10:
        scale = 0.5 * (short_side / max(sin, cos)) if max(sin, cos) else 0.0
        inner_w, inner_h = (
            (scale, scale) if width >= height else (scale, scale)
        )
    else:
        denominator = cos * cos - sin * sin
        inner_w = (width * cos - height * sin) / denominator
        inner_h = (height * cos - width * sin) / denominator
    inner_w = int(max(1, min(width, inner_w)))
    inner_h = int(max(1, min(height, inner_h)))
    x0 = (width - inner_w) // 2
    y0 = (height - inner_h) // 2
    return rotated[y0 : y0 + inner_h, x0 : x0 + inner_w]


def crop_border(image: np.ndarray, border: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = border
    height, width = image.shape[:2]
    return image[top : height - bottom if bottom else height,
                 left : width - right if right else width]


def neutral_luma(image: np.ndarray, headroom: float = 0.04) -> np.ndarray:
    """Collapse a cast-affected monochrome print to a clean neutral grey image.

    Every channel is stretched onto a shared percentile anchor, which removes
    the cast without the hue shifts a naive grey-world balance introduces, then
    the channels are averaged: on a monochrome original each one carries the
    same signal at a different gain, so averaging also lowers chroma noise.

    The stretch deliberately stops short of 0 and 255. Landing the anchors on
    the limits clips the shoulder of the print's tone curve, and every later
    stage -- sharpening, grading, colourisation -- then works from data that has
    already lost its highlight roll-off.
    """
    channels = []
    for index in range(3):
        channel = image[:, :, index].astype(np.float32)
        low, high = np.percentile(channel, (0.5, 99.5))
        if high - low < 1e-3:
            high = low + 1.0
        scaled = (channel - low) / (high - low)
        channels.append(np.clip(scaled * (1.0 - 2 * headroom) + headroom, 0.0, 1.0))
    return (np.mean(channels, axis=0) * 255.0).astype(np.uint8)


def denoise(grey: np.ndarray, strength: float) -> np.ndarray:
    """Edge-preserving grain removal; strength 0 returns the input unchanged."""
    if strength <= 0:
        return grey
    return cv2.fastNlMeansDenoising(
        grey, None, h=float(strength), templateWindowSize=7, searchWindowSize=21
    )


def prepare(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, dict]:
    """Run the full preparation stage.

    Returns the prepared neutral-grey BGR image plus a report of what was
    actually applied, so a run can be reproduced and audited.
    """
    report: dict = {"input_size": image.shape[1::-1]}

    angle = estimate_tilt(image, config.max_border_fraction) if config.auto_crop_frame else 0.0
    working = deskew(image, angle)
    report["deskew_degrees"] = round(angle, 3)

    if config.manual_crop is not None:
        border = config.manual_crop
    elif config.auto_crop_frame:
        border = detect_frame_border(working, config.max_border_fraction)
    else:
        border = (0, 0, 0, 0)
    working = crop_border(working, border)
    report["crop_trbl"] = tuple(int(value) for value in border)
    report["size_after_crop"] = working.shape[1::-1]

    grey = (
        neutral_luma(working)
        if config.neutralize_cast
        else cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    )
    report["cast_neutralized"] = config.neutralize_cast

    grey = denoise(grey, config.denoise_strength)
    report["denoise_h"] = config.denoise_strength
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), report
