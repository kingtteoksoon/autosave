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
#: The border is cropped to this percentile of its per-column depth.
_EDGE_DEPTH_PERCENTILE = 95.0
#: A column counts as frame down to the last depth at which this fraction of
#: the run is still off-chroma.
_EDGE_RUN_DENSITY = 0.60
#: Sanity floor on how far apart the two Otsu classes must sit.
_MIN_CLASS_GAP = 4.0
#: The frame's soft shadow edge extends a little past the colour step.
_EDGE_PAD = 6
#: A border search never grows past this fraction of the image.
_HARD_CAP_FRACTION = 0.35

_SIDES = ("top", "bottom", "left", "right")


def chroma_distance(image: np.ndarray) -> np.ndarray:
    """Per-pixel CIELAB chroma distance from the picture interior's median hue."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = lab[:, :, 1:] - 128.0
    height, width = image.shape[:2]
    interior = chroma[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    reference = np.median(interior.reshape(-1, 2), axis=0)
    return np.linalg.norm(chroma - reference, axis=-1)


def interior_level(distance: np.ndarray) -> float:
    """How far off-chroma the picture's own content gets, as a reference level.

    Used to read a band that Otsu cannot split: one uniform class is either all
    frame or all picture, and only a comparison against the picture's own
    spread says which.
    """
    height, width = distance.shape
    middle = distance[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return float(np.percentile(middle, 95))


def _side_band(distance: np.ndarray, side: str, limit: int) -> np.ndarray:
    """The edge strip for one side, oriented so row 0 is the outermost line."""
    if side == "top":
        return distance[:limit]
    if side == "bottom":
        return distance[::-1][:limit]
    if side == "left":
        return distance[:, :limit].T
    if side == "right":
        return distance[:, ::-1][:, :limit].T
    raise ValueError(f"unknown side {side!r}")


def _side_extent(distance: np.ndarray, side: str) -> int:
    """How far a band on this side may extend before running out of image."""
    height, width = distance.shape
    return height if side in ("top", "bottom") else width


def _band_cutoff(band: np.ndarray) -> tuple[float, float]:
    """Otsu split of one band into frame-coloured and picture pixels.

    The cutoff is computed per side rather than once for the whole border. A
    strongly coloured frame on one edge otherwise drags the shared threshold
    above a weaker border elsewhere -- a gilt frame at the top will hide the
    cloth a print is lying on at the bottom.
    """
    quantised = (np.clip(band / _CHROMA_SCALE, 0.0, 1.0) * 255).astype(np.uint8)
    level, _ = cv2.threshold(quantised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = float(level) / 255.0 * _CHROMA_SCALE
    low, high = band[band <= cutoff], band[band > cutoff]
    gap = float(high.mean() - low.mean()) if low.size and high.size else 0.0
    return cutoff, gap


def _run_depths(band: np.ndarray, cutoff: float) -> np.ndarray:
    """Per-column depth of the off-chroma border, measured from the outer edge.

    Density is evaluated over a short sliding window rather than cumulatively
    from the edge. A cumulative measure keeps a wide border alive well past its
    real edge -- a border covering 60% of the window still averages 60% -- so
    the reported depth would depend on how far the search happened to look.
    A local window makes the measurement scale-free while still tolerating the
    pale threads and ornament that break a strictly unbroken run.
    """
    above = (band > cutoff).astype(np.float32)
    window = max(9, (band.shape[0] // 20) | 1)
    local = cv2.blur(above, (1, window), borderType=cv2.BORDER_REPLICATE)
    dense = local >= _EDGE_RUN_DENSITY
    reversed_hit = np.argmax(dense[::-1], axis=0)
    depths = band.shape[0] - reversed_hit
    return np.where(dense.any(axis=0), depths, 0).astype(np.float32)


def _scan_side(
    distance: np.ndarray, side: str, max_fraction: float, picture_level: float
) -> int:
    """Pixels of frame on one side, or 0 if no boundary is found there.

    The search window grows while the off-chroma band still fills it, because a
    border wider than the initial window would otherwise be cropped to the
    window rather than to its real edge. A band that never ends, even at the
    hard cap, is picture content rather than a frame, so nothing is cropped:
    finding no boundary is a better answer than inventing one.

    The crop itself is taken from a high percentile of the per-column depth
    rather than from where half the line is still frame. On a border that runs
    at an angle to the sensor those differ by the whole width of the wedge, and
    the median leaves a triangle of frame in the picture.
    """
    extent = _side_extent(distance, side)
    limit = max(1, int(extent * max_fraction))
    cap = max(1, int(extent * _HARD_CAP_FRACTION))

    while True:
        band = cv2.GaussianBlur(_side_band(distance, side, limit), (0, 0), 2.0)
        cutoff, gap = _band_cutoff(band)

        inner = band[band <= cutoff]
        inner_level = float(inner.mean()) if inner.size else float(band.mean())
        if inner_level > max(picture_level * 1.5, picture_level + 4.0):
            # Even the quieter of the two classes is far off the picture's own
            # chroma, so this whole window is still border. Otsu has split the
            # border internally -- cloth against its woven stripe, frame against
            # its ornament -- and its cutoff means nothing here.
            head = tail = 1.0
        elif gap < _MIN_CLASS_GAP:
            # One uniform class that is not off-chroma: all picture, no border.
            return 0
        else:
            coverage = (band > cutoff).mean(axis=1)
            if coverage.size == 0:
                return 0
            head, tail = float(coverage[0]), float(coverage[-1])

        if head < _EDGE_PRESENCE:
            return 0
        if tail >= _EDGE_CONTINUATION:
            if limit >= cap:
                return 0
            limit = min(limit * 2, cap)
            continue

        depth = float(np.percentile(_run_depths(band, cutoff), _EDGE_DEPTH_PERCENTILE))
        return int(min(depth + _EDGE_PAD, band.shape[0]))


def detect_frame_border(
    image: np.ndarray, max_fraction: float = 0.10
) -> tuple[int, int, int, int]:
    """Pixels of picture frame to trim from each edge, as (top, bottom, left, right)."""
    distance = chroma_distance(image)
    level = interior_level(distance)
    return tuple(_scan_side(distance, side, max_fraction, level) for side in _SIDES)


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
    border = _scan_side(distance, "top", max_fraction, interior_level(distance))
    if border == 0:
        return 0.0

    # Look a little past the detected border so the boundary itself is inside
    # the window being measured.
    window = min(int(border * 1.5) + _EDGE_PAD, distance.shape[0])
    band = cv2.GaussianBlur(_side_band(distance, "top", window), (0, 0), 2.0)
    cutoff, _ = _band_cutoff(band)

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
