#!/usr/bin/env python3
"""CLI: restore old black-and-white photographs with local ONNX models.

Examples
--------
    python restore.py photo.jpg -o restored/
    python restore.py album/ -o restored/ --no-colorize --upscale 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from photo_restore import RestoreOptions, restore_path

SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.suffix.lower() in SUFFIXES)
        elif path.suffix.lower() in SUFFIXES:
            found.append(path)
        else:
            print(f"[skip] unsupported input: {path}", file=sys.stderr)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="AI restoration for old / black-and-white photos")
    parser.add_argument("inputs", nargs="+", help="image files or directories")
    parser.add_argument("-o", "--out-dir", default="restored", help="output directory (default: restored)")
    parser.add_argument("--face-model", default="codeformer", choices=["codeformer", "gfpgan", "gpen"])
    parser.add_argument("--fidelity", type=float, default=0.85,
                        help="CodeFormer w: 1.0 keeps the real face, lower is cleaner but less faithful")
    parser.add_argument("--face-blend", type=float, default=0.9, help="opacity of the restored face (0-1)")
    parser.add_argument("--upscale", type=int, default=2, choices=[1, 2, 4], help="1 disables super resolution")
    parser.add_argument("--neutralize", type=float, default=0.7,
                        help="strength of the global colour-cast correction applied to the colourised result")
    parser.add_argument("--saturation", type=float, default=0.85)
    parser.add_argument("--no-flip-tta", action="store_true",
                        help="skip the mirrored colour pass (faster, slightly less stable colour)")
    parser.add_argument("--denoise", type=int, default=5, help="non-local-means strength (0 disables)")
    parser.add_argument("--no-colorize", action="store_true", help="black-and-white restoration only")
    parser.add_argument("--no-crop", action="store_true", help="keep frame borders")
    parser.add_argument("--outputs", default="bw,color,comparison", help="comma separated: bw,color,comparison")
    args = parser.parse_args()

    images = collect(args.inputs)
    if not images:
        print("no input images found", file=sys.stderr)
        return 1

    options = RestoreOptions(
        crop_frame=not args.no_crop,
        denoise_strength=args.denoise,
        face_model=args.face_model,
        face_fidelity=args.fidelity,
        face_blend=args.face_blend,
        upscale=args.upscale,
        colorize=not args.no_colorize,
        neutralize=args.neutralize,
        flip_tta=not args.no_flip_tta,
        saturation=args.saturation,
        outputs=[o.strip() for o in args.outputs.split(",") if o.strip()],
    )

    for index, image in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image}", flush=True)
        report = restore_path(image, args.out_dir, options)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
