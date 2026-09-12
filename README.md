# autosave

## photo_restore

Restoration pipeline for old and black-and-white photographs, including
phone snapshots of framed prints. Runs entirely on CPU.

## What it does

| Stage | Purpose | Method |
| --- | --- | --- |
| prepare | remove the picture frame or the surface a print is lying on, correct capture tilt, neutralise the colour cast, denoise | CIELAB chroma segmentation with an adaptive search window, Theil-Sen edge fit, per-channel percentile balance, non-local means |
| defects | remove dust and scratches | residual-vs-median blobs restricted to locally smooth regions, Telea inpainting |
| faces | rebuild facial detail lost to blur and print grain | YOLOFace-8n landmarks, FFHQ-512 alignment, CodeFormer or GFPGAN |
| upscale | raise resolution and recover micro-detail | Real-ESRGAN, tiled with cosine-blended overlaps |
| grade | set black and white points, add contrast, sharpen | luminance-only levels, S-curve, unsharp mask |
| colorize | add colour to a monochrome original | DDColor chroma over the restored luminance |

Two files come out of every run: a monochrome master and a colourised version.
The monochrome master is the faithful restoration of a monochrome original;
colourisation is an informed guess laid on top of it.

## Install and run

```bash
pip install -r requirements.txt
python -m photo_restore input/photo_01.jpg -o output
python -m photo_restore input/ -o output        # a whole directory
```

Model weights (about 1.5 GB) download on first use into
`~/.cache/photo_restore`. Override the location with `PHOTO_RESTORE_MODELS`.

Each run writes `<name>_restored_bw.jpg`, `<name>_restored_color.jpg`, a
`<name>_comparison.jpg` contact sheet, and a `<name>_report.json` recording
every parameter, stage timing, and the SHA-256 of each model used.

## Useful options

```bash
--face-model codeformer|gfpgan|none   # CodeFormer keeps more identity, GFPGAN is smoother
--fidelity 0.7                        # CodeFormer: 0 invents freely, 1 stays literal
--face-blend 0.85                     # how much restored face to mix over the original
--colorizer none                      # monochrome only
--chroma 0.85                         # colour intensity
--upscale esrgan_x2|esrgan_x4|none
--no-crop                             # keep the frame and the tilt as shot
```

## Design decisions worth knowing

**Colourisation only ever adds chroma.** Luminance is taken from the graded
monochrome master and never touched by the coloriser, so colour can never
soften detail or shift exposure.

**Faces are restored before super resolution.** The restorers work on a fixed
512x512 aligned crop, so upscaling first would only hand them an already
hallucinated face.

**The restored face is blended, not substituted.** Generative restorers invent
plausible skin and eyes. On a family portrait, keeping a share of the true
texture is what preserves the likeness, which is why `face_blend` defaults to
0.85 rather than 1.0.

**Colouriser output is white balanced.** DDColor drifts magenta on studio
portraits. The drift is measured from the frame's brightest pixels, which are
normally near-neutral, and subtracted. Chroma is also rolled off in deep
shadows, where a coloriser's guesses are least reliable and real photographic
shadows desaturate anyway.

**The border search grows to fit the border.** A print photographed on a
table can sit on a wide expanse of cloth, far past the 10% of the frame the
search starts with. The window doubles while the off-chroma band still fills
it, and a band that never ends within the hard cap is read as picture content,
so nothing is cropped. Depth is then taken from a high percentile of the
per-column border thickness rather than from where half the line is still
border: on a border running at an angle to the sensor those differ by the whole
width of the wedge, and a median crop leaves a triangle of cloth in the picture.
Density along each column is measured over a short sliding window, so a woven
stripe or a run of pale ornament does not cut the measurement short.

**Dust removal limits itself.** Damage is rare. When the detector flags more
than 1% of the frame it has caught the print's own grain, so it raises its
threshold until what remains is plausibly damage. Inpainting grain would smooth
real texture out of the whole picture; tempering grain is the denoiser's job.

**The white balance correction is capped hard.** The white-patch assumption
holds when the brightest surface is a white collar and fails when it is a large
lit backdrop, where it returns a big shift that would drag skin far off
natural. A small measured tint is worth removing; a large one is evidence the
assumption does not hold, so the correction can nudge but never dominate.

**CLAHE is off by default.** On a portrait the background is one large smooth
vignette, and equalising it tile by tile lifts it to mid grey and destroys the
depth the photographer lit for. Grading is global instead.

**DeOldify is not supported.** Its ONNX export was evaluated and rejected: the
output tensor uses an undocumented range that could not be denormalised to
chroma reliably.

## Tests

```bash
python -m pytest tests -q
```

The neural stages need the full weight set, so they are covered by running the
CLI. Everything deterministic is unit tested on synthetic inputs whose correct
answer is known by construction.

## Model credits

Weights are the ONNX exports published by the
[FaceFusion assets](https://github.com/facefusion/facefusion-assets) release:
YOLOFace-8n, GFPGAN 1.4, CodeFormer, Real-ESRGAN, DDColor. Each model's
original licence applies to its own use.
