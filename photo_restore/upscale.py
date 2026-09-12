"""Tiled Real-ESRGAN super resolution.

The network is fully convolutional but memory grows with the square of the
input, so full frames are processed as overlapping tiles. Overlaps are blended
with a cosine ramp rather than hard-cut, which removes the seam artefacts a
naive grid produces on smooth gradients such as a studio backdrop.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import runtime
from .config import RestoreConfig


def _ramp(length: int, overlap: int) -> np.ndarray:
    """Weight profile along one axis: cosine ramp-in, flat, cosine ramp-out."""
    weight = np.ones(length, dtype=np.float32)
    if overlap > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, min(overlap, length))))
        weight[: ramp.size] = np.minimum(weight[: ramp.size], ramp)
        weight[-ramp.size :] = np.minimum(weight[-ramp.size :], ramp[::-1])
    return weight


def _infer(session, tile: np.ndarray) -> np.ndarray:
    """Run one BGR uint8 tile through the network and return BGR float output."""
    rgb = tile[:, :, ::-1].astype(np.float32) / 255.0
    output = session.run(None, {"input": runtime.to_nchw(rgb)})[0]
    return np.clip(runtime.from_nchw(output), 0.0, 1.0)[:, :, ::-1] * 255.0


def upscale(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, dict]:
    """Super-resolve the frame, then cap its long edge at the configured size."""
    report: dict = {"model": config.upscale_model}
    if config.upscale_model == "none":
        report["scale"] = 1
        return image, report

    session = runtime.load(config.upscale_model, config.intra_op_threads)
    height, width = image.shape[:2]
    probe = _infer(session, image[:32, :32])
    scale = probe.shape[0] // 32
    report["scale"] = scale

    tile = max(64, config.upscale_tile)
    overlap = max(0, min(config.upscale_overlap, tile // 4))
    step = tile - overlap
    accumulator = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    weights = np.zeros((height * scale, width * scale, 1), dtype=np.float32)

    for y in range(0, height, step):
        for x in range(0, width, step):
            y0, x0 = min(y, max(0, height - tile)), min(x, max(0, width - tile))
            patch = image[y0 : y0 + tile, x0 : x0 + tile]
            result = _infer(session, patch)
            mask = (
                _ramp(result.shape[0], overlap * scale)[:, None]
                * _ramp(result.shape[1], overlap * scale)[None, :]
            )[:, :, None]
            ty, tx = y0 * scale, x0 * scale
            accumulator[ty : ty + result.shape[0], tx : tx + result.shape[1]] += result * mask
            weights[ty : ty + result.shape[0], tx : tx + result.shape[1]] += mask
            if x0 == max(0, width - tile):
                break
        if y0 == max(0, height - tile):
            break

    merged = (accumulator / np.maximum(weights, 1e-6)).clip(0, 255).astype(np.uint8)
    report["size_after_upscale"] = merged.shape[1::-1]

    long_edge = max(merged.shape[:2])
    if config.max_long_edge and long_edge > config.max_long_edge:
        factor = config.max_long_edge / long_edge
        merged = cv2.resize(
            merged,
            (round(merged.shape[1] * factor), round(merged.shape[0] * factor)),
            interpolation=cv2.INTER_AREA,
        )
        report["resampled_to"] = merged.shape[1::-1]
    return merged, report
