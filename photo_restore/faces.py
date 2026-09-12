"""Face detection, alignment, and generative face restoration.

Detection uses YOLOFace-8n, which returns a box plus the five canonical
landmarks in one pass. The landmarks drive a similarity warp onto the FFHQ
512x512 template that GFPGAN and CodeFormer were trained on; restoring an
unaligned crop is the single most common cause of melted output.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import runtime
from .config import RestoreConfig

#: FFHQ 512 alignment template (right eye, left eye, nose, right mouth, left mouth).
FFHQ_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)
CROP_SIZE = 512
DETECTOR_SIZE = 640


@dataclass
class Face:
    """A detected face: pixel-space box, five landmarks, detector confidence."""

    box: np.ndarray
    landmarks: np.ndarray
    score: float


def _prepare_detector_frame(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the frame inside the detector square, top-left aligned, and normalise."""
    height, width = image.shape[:2]
    scale = min(DETECTOR_SIZE / height, DETECTOR_SIZE / width)
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((DETECTOR_SIZE, DETECTOR_SIZE, 3), dtype=np.float32)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    canvas = (canvas - 127.5) / 128.0
    return runtime.to_nchw(canvas), scale


def detect_faces(image: np.ndarray, config: RestoreConfig) -> list[Face]:
    """Detect faces, largest first, above the configured score threshold."""
    session = runtime.load("face_detector", config.intra_op_threads)
    tensor, scale = _prepare_detector_frame(image)
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detection = np.squeeze(raw).T  # (8400, 20)

    boxes_raw, scores_raw, landmarks_raw = np.split(detection, [4, 5], axis=1)
    scores = scores_raw.ravel()
    keep = np.where(scores > config.face_score_threshold)[0]
    if keep.size == 0:
        return []

    boxes_raw, landmarks_raw, scores = boxes_raw[keep], landmarks_raw[keep], scores[keep]
    centres, sizes = boxes_raw[:, :2], boxes_raw[:, 2:]
    corners = np.concatenate((centres - sizes / 2, centres + sizes / 2), axis=1) / scale
    landmarks = landmarks_raw.reshape(-1, 5, 3)[:, :, :2] / scale

    indices = cv2.dnn.NMSBoxes(
        [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in corners],
        scores.tolist(),
        config.face_score_threshold,
        0.4,
    )
    order = np.array(indices).ravel() if len(indices) else np.arange(len(scores))
    faces = [
        Face(corners[i], landmarks[i].astype(np.float32), float(scores[i])) for i in order
    ]
    faces.sort(key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), reverse=True)
    return faces


def align(image: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp a face onto the FFHQ template; returns the crop and its affine matrix."""
    template = FFHQ_TEMPLATE * CROP_SIZE
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks, template, method=cv2.RANSAC, ransacReprojThreshold=100
    )
    crop = cv2.warpAffine(
        image, matrix, (CROP_SIZE, CROP_SIZE), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA
    )
    return crop, matrix


def _feather_mask(blur: float = 0.3, padding: float = 0.06) -> np.ndarray:
    """Soft-edged box mask so the restored crop dissolves into the original."""
    mask = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    inset = max(1, int(CROP_SIZE * padding))
    mask[:inset, :] = 0.0
    mask[-inset:, :] = 0.0
    mask[:, :inset] = 0.0
    mask[:, -inset:] = 0.0
    amount = max(1.0, CROP_SIZE * 0.5 * blur * 0.25)
    return cv2.GaussianBlur(mask, (0, 0), amount)


def paste_back(
    image: np.ndarray, crop: np.ndarray, matrix: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Blend a restored 512x512 crop back into full-resolution space."""
    inverse = cv2.invertAffineTransform(matrix)
    size = image.shape[1::-1]
    warped = cv2.warpAffine(crop, inverse, size, borderMode=cv2.BORDER_REPLICATE)
    alpha = np.clip(cv2.warpAffine(mask, inverse, size), 0.0, 1.0)[:, :, None]
    return (alpha * warped.astype(np.float32) + (1 - alpha) * image.astype(np.float32)).astype(
        np.uint8
    )


def _to_model_tensor(crop: np.ndarray) -> np.ndarray:
    """BGR uint8 crop -> RGB NCHW tensor scaled to [-1, 1]."""
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    return runtime.to_nchw((rgb - 0.5) / 0.5)


def _from_model_tensor(tensor: np.ndarray) -> np.ndarray:
    """Model output in [-1, 1] -> BGR uint8 crop."""
    frame = np.clip(runtime.from_nchw(tensor), -1.0, 1.0)
    return np.rint((frame + 1.0) / 2.0 * 255.0).astype(np.uint8)[:, :, ::-1]


def restore_crop(crop: np.ndarray, config: RestoreConfig) -> np.ndarray:
    """Run the configured generative restorer over one aligned 512x512 crop."""
    session = runtime.load(config.face_model, config.intra_op_threads)
    inputs = {"input": _to_model_tensor(crop)}
    if any(node.name == "weight" for node in session.get_inputs()):
        # CodeFormer fidelity: 0 favours invention, 1 stays close to the input.
        inputs["weight"] = np.array(config.codeformer_fidelity, dtype=np.float64)
    output = session.run(None, inputs)[0]
    return _from_model_tensor(output)


def is_monochrome(image: np.ndarray, tolerance: float = 2.0) -> bool:
    """True when the frame carries no real colour, within a small tolerance."""
    sample = image[::4, ::4].astype(np.float32)
    return float(np.mean(np.abs(sample - sample.mean(axis=2, keepdims=True)))) < tolerance


def restore_faces(image: np.ndarray, config: RestoreConfig) -> tuple[np.ndarray, list[dict]]:
    """Detect, restore and blend every face in the frame.

    The restored crop is mixed back at ``face_blend`` rather than replacing the
    original outright: generative restorers reconstruct plausible skin and eyes,
    and on a family portrait keeping some of the true texture is what preserves
    the likeness.
    """
    if config.face_model == "none":
        return image, []

    faces = detect_faces(image, config)
    mask = _feather_mask()
    monochrome = is_monochrome(image)
    result = image
    report: list[dict] = []
    for face in faces:
        crop, matrix = align(result, face.landmarks)
        restored = restore_crop(crop, config)
        if monochrome:
            # The restorers are trained on colour and tint their output slightly.
            # On a monochrome source that tint is an artefact, not information.
            restored = cv2.cvtColor(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(
            restored, config.face_blend, crop, 1.0 - config.face_blend, 0.0
        )
        result = paste_back(result, blended, matrix, mask)
        report.append(
            {
                "score": round(face.score, 3),
                "box": [int(v) for v in face.box],
                "model": config.face_model,
            }
        )
    return result, report
