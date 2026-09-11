"""Tiled Real-ESRGAN super resolution, sized for CPU-only machines."""
from __future__ import annotations

import cv2
import numpy as np

from .models import session

SCALERS = {2: "real_esrgan_x2.onnx", 4: "real_esrgan_x4.onnx"}


def upscale(bgr: np.ndarray, scale: int = 2, tile: int = 256, overlap: int = 24) -> np.ndarray:
    """Upscale with overlapping tiles blended by a cosine ramp (no seams).

    Tiling keeps peak memory flat, which is what makes a 3 MP portrait feasible
    on a 4-core CPU box.
    """
    if scale not in SCALERS:
        raise ValueError(f"scale must be one of {sorted(SCALERS)}")
    net = session(SCALERS[scale])
    height, width = bgr.shape[:2]
    output = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    weights = np.zeros((height * scale, width * scale, 1), dtype=np.float32)
    step = tile - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):
            y1, x1 = min(y + tile, height), min(x + tile, width)
            y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
            patch = bgr[y0:y1, x0:x1]
            tensor = np.expand_dims((patch[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1), 0)
            result = net.run(None, {"input": tensor})[0][0]
            result = np.clip(result.transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255.0

            ramp = _ramp(result.shape[0], result.shape[1], overlap * scale)
            output[y0 * scale : y1 * scale, x0 * scale : x1 * scale] += result * ramp
            weights[y0 * scale : y1 * scale, x0 * scale : x1 * scale] += ramp
            if x1 >= width:
                break
        if y1 >= height:
            break
    return np.clip(output / np.maximum(weights, 1e-6), 0, 255).astype(np.uint8)


def _ramp(height: int, width: int, feather: int) -> np.ndarray:
    """Separable cosine window used to cross-fade neighbouring tiles."""
    def profile(length: int) -> np.ndarray:
        window = np.ones(length, dtype=np.float32)
        edge = max(1, min(feather, length // 2))
        fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge, dtype=np.float32))
        window[:edge] = fade
        window[-edge:] = fade[::-1]
        return window
    return (profile(height)[:, None] * profile(width)[None, :])[..., None]
