"""Command line entry point: ``python -m photo_restore``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .config import MODEL_ASSETS, RestoreConfig, file_digest, model_path
from .pipeline import comparison_sheet, restore, save

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect_inputs(paths: list[str]) -> list[Path]:
    """Expand files and directories into a sorted list of image paths."""
    found: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            found.extend(
                child for child in sorted(path.iterdir())
                if child.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
        else:
            raise SystemExit(f"not an image: {path}")
    if not found:
        raise SystemExit("no input images found")
    return found


def build_parser() -> argparse.ArgumentParser:
    defaults = RestoreConfig()
    parser = argparse.ArgumentParser(
        prog="photo_restore",
        description="Restore and colourise old or black-and-white photographs.",
    )
    parser.add_argument("inputs", nargs="+", help="image files or directories")
    parser.add_argument("-o", "--output", default="output", help="output directory")
    parser.add_argument(
        "--face-model", default=defaults.face_model, choices=["codeformer", "gfpgan", "none"]
    )
    parser.add_argument("--fidelity", type=float, default=defaults.codeformer_fidelity,
                        help="CodeFormer fidelity, 0 generative .. 1 faithful")
    parser.add_argument("--face-blend", type=float, default=defaults.face_blend,
                        help="weight of the restored face over the original")
    parser.add_argument("--upscale", default=defaults.upscale_model,
                        choices=["esrgan_x2", "esrgan_x4", "none"])
    parser.add_argument("--max-long-edge", type=int, default=defaults.max_long_edge)
    parser.add_argument("--colorizer", default=defaults.colorizer, choices=["ddcolor", "none"])
    parser.add_argument("--chroma", type=float, default=defaults.chroma_strength)
    parser.add_argument("--denoise", type=float, default=defaults.denoise_strength)
    parser.add_argument("--no-crop", action="store_true", help="keep frame and tilt as shot")
    parser.add_argument("--threads", type=int, default=defaults.intra_op_threads)
    parser.add_argument("--no-sheet", action="store_true", help="skip the comparison sheet")
    return parser


def config_from_args(args: argparse.Namespace) -> RestoreConfig:
    return RestoreConfig(
        auto_crop_frame=not args.no_crop,
        denoise_strength=args.denoise,
        face_model=args.face_model,
        codeformer_fidelity=args.fidelity,
        face_blend=args.face_blend,
        upscale_model=args.upscale,
        max_long_edge=args.max_long_edge,
        colorizer=args.colorizer,
        chroma_strength=args.chroma,
        intra_op_threads=args.threads,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    inputs = collect_inputs(args.inputs)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in inputs:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skipping unreadable file: {source}", file=sys.stderr)
            continue
        print(f"restoring {source} ({image.shape[1]}x{image.shape[0]}) ...", flush=True)

        master, coloured, report = restore(image, config)
        stem = source.stem
        report["source"] = str(source)
        report["outputs"] = {
            "bw": str(save(out_dir / f"{stem}_restored_bw.jpg", master, config.jpeg_quality)),
            "color": str(save(out_dir / f"{stem}_restored_color.jpg", coloured, config.jpeg_quality)),
        }
        if not args.no_sheet:
            sheet = comparison_sheet(image, master, coloured)
            report["outputs"]["sheet"] = str(
                save(out_dir / f"{stem}_comparison.jpg", sheet, 92)
            )
        report["model_digests"] = {
            name: file_digest(model_path(name))[:16]
            for name in sorted(MODEL_ASSETS)
            if name in {config.face_model, config.upscale_model, config.colorizer, "face_detector"}
        }
        (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, default=str))
        timings = report["timings_seconds"]
        print(
            f"  done in {sum(timings.values()):.1f}s -> {report['output_size'][0]}x"
            f"{report['output_size'][1]}  ({', '.join(f'{k} {v}s' for k, v in timings.items())})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
