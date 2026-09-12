"""Model registry and pipeline configuration.

Weights are ONNX exports published as GitHub release assets, so the pipeline runs
on CPU through onnxruntime with no PyTorch/CUDA dependency.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ASSET_BASE = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
)

#: Logical model name -> release asset file name.
MODEL_ASSETS: dict[str, str] = {
    "face_detector": "yoloface_8n.onnx",
    "gfpgan": "gfpgan_1.4.onnx",
    "codeformer": "codeformer.onnx",
    "ddcolor": "ddcolor.onnx",
    "esrgan_x2": "real_esrgan_x2.onnx",
    "esrgan_x4": "real_esrgan_x4.onnx",
}

DEFAULT_MODEL_DIR = Path(
    os.environ.get("PHOTO_RESTORE_MODELS", Path.home() / ".cache" / "photo_restore")
)


@dataclass
class RestoreConfig:
    """Tunable knobs for a single restoration run.

    The defaults are tuned for a phone photograph of a framed monochrome print:
    conservative face blending (identity preservation beats plastic perfection)
    and gentle grading.
    """

    # --- source preparation -------------------------------------------------
    auto_crop_frame: bool = True
    max_border_fraction: float = 0.10
    manual_crop: tuple[int, int, int, int] | None = None  # (top, bottom, left, right)
    neutralize_cast: bool = True
    denoise_strength: float = 6.0  # 0 disables fastNlMeans denoising
    remove_defects: bool = True
    defect_threshold: float = 14.0  # grey levels a speck must stand out by
    defect_smooth_percentile: float = 85.0  # how much of the frame counts as smooth
    defect_max_area_fraction: float = 3e-5  # largest blob still treated as a defect
    defect_max_frame_fraction: float = 0.01  # of the frame; above this it is grain, not dust

    # --- face restoration ---------------------------------------------------
    face_model: str = "codeformer"  # "codeformer" | "gfpgan" | "none"
    codeformer_fidelity: float = 0.7  # 0 = most generative, 1 = closest to input
    face_blend: float = 0.85  # weight of the restored face over the original
    face_score_threshold: float = 0.35

    # --- super resolution ---------------------------------------------------
    upscale_model: str = "esrgan_x2"  # "esrgan_x2" | "esrgan_x4" | "none"
    upscale_tile: int = 256
    upscale_overlap: int = 32
    max_long_edge: int = 4200  # downsample after upscaling to cap output size

    # --- colorization -------------------------------------------------------
    colorizer: str = "ddcolor"  # "ddcolor" | "none"
    colorizer_size: int = 512
    chroma_strength: float = 0.85  # scales predicted a/b channels
    chroma_refine_radius: float = 0.015  # guided-filter radius as a fraction of width
    white_balance: bool = True  # neutralise a global tint in the predicted chroma
    white_balance_strength: float = 0.8
    white_balance_limit: float = 5.0  # largest a/b shift the correction may apply
    shadow_chroma_floor: float = 0.45  # chroma retained in the deepest shadows
    shadow_chroma_knee: float = 0.45  # lightness fraction above which chroma is untouched

    # --- grading ------------------------------------------------------------
    tone_contrast: float = 0.20  # global S-curve strength, 0 disables
    black_point: float = 3.0  # target level for the darkest anchor percentile
    white_point: float = 246.0  # target level for the brightest anchor percentile
    clahe_clip: float = 0.0  # local contrast; off by default, it flattens vignettes
    unsharp_amount: float = 0.45
    unsharp_radius: float = 1.6
    jpeg_quality: int = 96

    intra_op_threads: int = field(default_factory=lambda: min(8, os.cpu_count() or 4))


def model_path(name: str, model_dir: Path | None = None) -> Path:
    """Return the local path for ``name``, downloading the asset if missing."""
    if name not in MODEL_ASSETS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_ASSETS)}")
    directory = Path(model_dir or DEFAULT_MODEL_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MODEL_ASSETS[name]
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{ASSET_BASE}/{MODEL_ASSETS[name]}"
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(target)
    return target


def file_digest(path: Path) -> str:
    """SHA-256 of a file, used to record exactly which weights produced a result."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
