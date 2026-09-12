"""Face pipeline: YOLOFace detection -> FFHQ alignment -> blind face restoration.

Faces carry the identity of an old portrait, so they are restored in their own
512x512 aligned space (where the networks were trained) and blended back with a
feathered mask and luminance matching to avoid a visible patch.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import session

# Standard FFHQ 512 five-point template (left eye, right eye, nose, mouth corners).
FFHQ_TEMPLATE = np.array(
    [[0.37691676, 0.46864664],
     [0.62285697, 0.46609768],
     [0.50123859, 0.61331904],
     [0.39308822, 0.72541100],
     [0.61150205, 0.72490465]], dtype=np.float32) * 512.0

RESTORERS = {"codeformer": "codeformer.onnx", "gfpgan": "gfpgan_1.4.onnx", "gpen": "gpen_bfr_512.onnx"}


@dataclass
class Face:
    score: float
    box: np.ndarray      # x1, y1, x2, y2 in source pixels
    landmarks: np.ndarray  # (5, 2) in source pixels


def detect(bgr: np.ndarray, score_threshold: float = 0.3, iou_threshold: float = 0.4) -> list[Face]:
    """YOLOFace-8n detection returning boxes plus the 5 landmarks used for alignment."""
    height, width = bgr.shape[:2]
    ratio = min(640 / height, 640 / width)
    resized = cv2.resize(bgr, (int(round(width * ratio)), int(round(height * ratio))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((640, 640, 3), dtype=np.float32)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    blob = np.expand_dims(((canvas - 127.5) / 128.0).transpose(2, 0, 1), axis=0).astype(np.float32)

    raw = np.squeeze(session("yoloface_8n.onnx").run(None, {"input": blob})[0]).T  # (8400, 20)
    raw = raw[raw[:, 4] > score_threshold]

    faces: list[Face] = []
    for row in raw:
        cx, cy, w, h, score = row[:5]
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32) / ratio
        landmarks = (row[5:].reshape(5, 3)[:, :2] / ratio).astype(np.float32)
        faces.append(Face(float(score), box, landmarks))

    faces.sort(key=lambda f: -f.score)
    kept: list[Face] = []
    for face in faces:
        if all(_iou(face.box, other.box) < iou_threshold for other in kept):
            kept.append(face)
    return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def align(bgr: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-warp a face into the 512x512 FFHQ frame the restorers expect."""
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, FFHQ_TEMPLATE, method=cv2.LMEDS)
    crop = cv2.warpAffine(bgr, matrix, (512, 512), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix


def restore_crop(crop_bgr: np.ndarray, model: str = "codeformer", fidelity: float = 0.85) -> np.ndarray:
    """Run one 512x512 aligned face through the chosen restoration network.

    ``fidelity`` is CodeFormer's ``w``: 1.0 stays closest to the real face,
    lower values hallucinate a cleaner but less faithful one.
    """
    net = session(RESTORERS[model])
    tensor = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.expand_dims(((tensor - 0.5) / 0.5).transpose(2, 0, 1), axis=0).astype(np.float32)
    feeds = {"input": tensor}
    if model == "codeformer":
        feeds["weight"] = np.array(fidelity, dtype=np.float64)
    output = net.run(None, feeds)[0][0]
    output = ((np.clip(output, -1, 1) + 1) / 2).transpose(1, 2, 0)[:, :, ::-1]
    return (output * 255).round().astype(np.uint8)


def _blend_mask(border: float = 0.08, blur: float = 0.11) -> np.ndarray:
    """Feathered 512x512 mask so the pasted face fades into the original print."""
    mask = np.zeros((512, 512), dtype=np.float32)
    inset = int(512 * border)
    mask[inset:-inset, inset:-inset] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), 512 * blur)


def _match_luma(restored: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match the restored crop's mean/contrast to the source so no seam shows."""
    weights = mask[..., None]
    total = float(weights.sum()) + 1e-6
    restored_f, reference_f = restored.astype(np.float32), reference.astype(np.float32)
    src_mean = (restored_f * weights).sum(axis=(0, 1)) / total
    ref_mean = (reference_f * weights).sum(axis=(0, 1)) / total
    src_std = np.sqrt(((restored_f - src_mean) ** 2 * weights).sum(axis=(0, 1)) / total) + 1e-6
    ref_std = np.sqrt(((reference_f - ref_mean) ** 2 * weights).sum(axis=(0, 1)) / total) + 1e-6
    scale = np.clip(ref_std / src_std, 0.75, 1.35)
    return np.clip((restored_f - src_mean) * scale + ref_mean, 0, 255)


def restore_faces(bgr: np.ndarray, *, model: str = "codeformer", fidelity: float = 0.85,
                  blend: float = 0.9, min_size: int = 64) -> tuple[np.ndarray, list[Face]]:
    """Detect, restore and composite every face found in the image."""
    faces = [f for f in detect(bgr) if (f.box[2] - f.box[0]) >= min_size]
    if not faces:
        return bgr, []

    result = bgr.astype(np.float32)
    mask = _blend_mask()
    height, width = bgr.shape[:2]
    for face in faces:
        crop, matrix = align(bgr, face.landmarks)
        restored = restore_crop(crop, model=model, fidelity=fidelity)
        restored = _match_luma(restored, crop, mask)
        restored = crop.astype(np.float32) * (1 - blend) + restored * blend

        inverse = cv2.invertAffineTransform(matrix)
        pasted = cv2.warpAffine(restored, inverse, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        pasted_mask = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        result = result * (1 - pasted_mask) + pasted * pasted_mask
    return np.clip(result, 0, 255).astype(np.uint8), faces
