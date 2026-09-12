"""Thin onnxruntime wrapper: cached CPU sessions with deterministic threading."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import model_path


@lru_cache(maxsize=None)
def _session(path: str, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider"]
    )


def load(name: str, threads: int = 4, model_dir: Path | None = None) -> ort.InferenceSession:
    """Load (and cache) the ONNX session for a logical model name."""
    return _session(str(model_path(name, model_dir)), threads)


def to_nchw(image: np.ndarray) -> np.ndarray:
    """HWC float array -> contiguous NCHW batch of one."""
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def from_nchw(tensor: np.ndarray) -> np.ndarray:
    """NCHW (or CHW) model output -> HWC float array."""
    array = np.asarray(tensor)
    if array.ndim == 4:
        array = array[0]
    return array.transpose(1, 2, 0)
