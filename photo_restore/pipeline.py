"""End-to-end restoration pipeline.

Order matters: physical clean-up, then identity-critical face restoration, then
super resolution, and only then colourisation - colour models read luminance,
so they benefit from every earlier step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import colorize as colorize_module
from . import faces as faces_module
from . import preprocess, upscale as upscale_module


@dataclass
class RestoreOptions:
    crop_frame: bool = True
    denoise_strength: int = 5
    face_model: str = "codeformer"
    face_fidelity: float = 0.85     # CodeFormer w: 1.0 = most faithful to the real face
    face_blend: float = 0.9
    upscale: int = 2                # 1 disables super resolution
    colorize: bool = True          # only applied to monochrome originals, see force_colorize
    force_colorize: bool = False   # re-colourise a photo that already has colour
    saturation: float = 0.85
    neutralize: float = 0.7   # pulls DDColor's global chroma bias back to neutral
    flip_tta: bool = True     # average the colour prediction with its mirrored pass
    jpeg_quality: int = 96
    outputs: list[str] = field(default_factory=lambda: ["bw", "color", "comparison"])


def restore_image(bgr: np.ndarray, options: RestoreOptions | None = None) -> dict:
    """Restore one already-loaded image; returns named BGR results plus a report."""
    options = options or RestoreOptions()
    report: dict = {"stages": {}}
    clock = time.time()

    prepared, prep_report = preprocess.prepare(
        bgr, do_crop=options.crop_frame, denoise_strength=options.denoise_strength)
    report.update(prep_report)
    report["stages"]["preprocess"] = round(time.time() - clock, 1)

    clock = time.time()
    restored, detected = faces_module.restore_faces(
        prepared, model=options.face_model, fidelity=options.face_fidelity, blend=options.face_blend)
    report["faces"] = [dict(score=round(f.score, 3), box=[int(v) for v in f.box]) for f in detected]
    report["stages"]["faces"] = round(time.time() - clock, 1)

    # Chroma is predicted before super resolution: the colour network is calibrated on
    # camera-resolution luminance, and SR shifts those statistics enough to skew the hues.
    chroma = None
    colorize_now = options.colorize and (report.get("is_monochrome") or options.force_colorize)
    report["colorized"] = bool(colorize_now)
    if options.colorize and not colorize_now:
        report["colorize_skipped"] = "original already has colour; pass force_colorize to override"
    if colorize_now:
        clock = time.time()
        chroma = colorize_module.neutralized_chroma(restored, neutralize=options.neutralize,
                                                   flip_tta=options.flip_tta)
        report["stages"]["colorize"] = round(time.time() - clock, 1)

    if options.upscale > 1:
        clock = time.time()
        restored = upscale_module.upscale(restored, scale=options.upscale)
        report["stages"]["upscale"] = round(time.time() - clock, 1)

    # Super resolution can leave a faint tint on a monochrome print; re-neutralise.
    if report.get("is_monochrome"):
        restored = cv2.cvtColor(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    restored = preprocess.unsharp(restored, amount=0.25)

    results = {"bw": restored}
    if chroma is not None:
        results["color"] = colorize_module.apply_chroma(restored, chroma, saturation=options.saturation)

    reference = cv2.resize(bgr, restored.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    side_by_side = results.get("color", restored)
    results["comparison"] = np.hstack([reference, side_by_side])
    report["output_size"] = restored.shape[1::-1]
    return {"results": results, "report": report}


def restore_path(src: str | Path, out_dir: str | Path, options: RestoreOptions | None = None) -> dict:
    """Restore one file on disk and write the requested outputs next to it."""
    options = options or RestoreOptions()
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outcome = restore_image(preprocess.load_bgr(str(src)), options)
    written = []
    for name in options.outputs:
        image = outcome["results"].get(name)
        if image is None:
            continue
        target = out_dir / f"{src.stem}_{name}.jpg"
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality])
        written.append(str(target))
    outcome["report"]["written"] = written
    return outcome["report"]
