"""Model zoo: on-demand download and cached ONNX Runtime sessions.

All weights are public ONNX exports published as GitHub release assets, so the
pipeline never needs a GPU, an API key, or a Hugging Face token.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import onnxruntime as ort

ASSET_BASE = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"

MODELS = {
    "yoloface_8n.onnx": f"{ASSET_BASE}/yoloface_8n.onnx",          # face detection + 5 landmarks
    "codeformer.onnx": f"{ASSET_BASE}/codeformer.onnx",            # blind face restoration
    "gfpgan_1.4.onnx": f"{ASSET_BASE}/gfpgan_1.4.onnx",            # alternative face restoration
    "gpen_bfr_512.onnx": f"{ASSET_BASE}/gpen_bfr_512.onnx",        # alternative face restoration
    "ddcolor.onnx": f"{ASSET_BASE}/ddcolor.onnx",                  # colourisation
    "real_esrgan_x2.onnx": f"{ASSET_BASE}/real_esrgan_x2.onnx",    # super resolution
    "real_esrgan_x4.onnx": f"{ASSET_BASE}/real_esrgan_x4.onnx",
}

DEFAULT_CACHE = Path(os.environ.get("PHOTO_RESTORE_MODELS", Path.home() / ".cache" / "photo-restore"))

_SESSIONS: dict[str, ort.InferenceSession] = {}


def model_path(name: str, cache_dir: Path | None = None) -> Path:
    """Return the local path of ``name``, downloading it on first use."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    cache = Path(cache_dir or DEFAULT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if not target.exists() or target.stat().st_size == 0:
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"[models] downloading {name} ...", flush=True)
        urllib.request.urlretrieve(MODELS[name], tmp)
        tmp.replace(target)
    return target


def session(name: str, cache_dir: Path | None = None, threads: int = 0) -> ort.InferenceSession:
    """Cached CPU inference session; sessions are reused across images."""
    if name not in _SESSIONS:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or (os.cpu_count() or 4)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        _SESSIONS[name] = ort.InferenceSession(
            str(model_path(name, cache_dir)), options, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[name]
