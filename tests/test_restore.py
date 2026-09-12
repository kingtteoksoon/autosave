"""Unit tests for the model-free parts of the pipeline.

The neural stages need 1.5 GB of weights, so they are exercised by running the
CLI rather than by unit tests. Everything deterministic -- geometry, tone,
chroma handling, defect detection -- is covered here on synthetic inputs whose
correct answer is known by construction.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from photo_restore import colorize, defects, grade, prep
from photo_restore.cli import collect_inputs
from photo_restore.config import RestoreConfig
from photo_restore.upscale import _ramp


def framed_photo(border: int = 40, size: tuple[int, int] = (600, 400)) -> np.ndarray:
    """A grey picture inside a saturated gold border, as a phone would capture it."""
    height, width = size
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    image[:, :, 0] = 130  # a mild blue cast over the picture area
    rng = np.random.default_rng(0)
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    image[:border, :] = (20, 150, 210)  # gold: strongly off-chroma
    return image


class TestFrameDetection:
    def test_finds_the_border_it_was_given(self) -> None:
        top, bottom, left, right = prep.detect_frame_border(framed_photo(border=40))
        assert 40 <= top <= 60
        assert (bottom, left, right) == (0, 0, 0)

    def test_unframed_photo_is_left_alone(self) -> None:
        rng = np.random.default_rng(1)
        plain = np.clip(rng.normal(120, 10, (600, 400, 3)), 0, 255).astype(np.uint8)
        assert prep.detect_frame_border(plain) == (0, 0, 0, 0)

    def test_crop_removes_exactly_the_requested_bands(self) -> None:
        image = np.zeros((100, 80, 3), dtype=np.uint8)
        assert prep.crop_border(image, (10, 5, 4, 6)).shape == (85, 70, 3)


class TestTilt:
    @pytest.mark.parametrize("angle", [-3.0, -1.5, 1.5, 3.0])
    def test_recovers_a_known_rotation(self, angle: float) -> None:
        image = framed_photo(border=50, size=(900, 600))
        centre = (image.shape[1] / 2, image.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(centre, -angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, image.shape[1::-1], borderMode=cv2.BORDER_REPLICATE
        )
        assert prep.estimate_tilt(rotated) == pytest.approx(angle, abs=0.6)

    def test_declines_to_rotate_without_a_frame(self) -> None:
        rng = np.random.default_rng(2)
        plain = np.clip(rng.normal(120, 10, (600, 400, 3)), 0, 255).astype(np.uint8)
        assert prep.estimate_tilt(plain) == 0.0

    def test_deskew_is_a_no_op_below_the_threshold(self) -> None:
        image = framed_photo()
        assert prep.deskew(image, 0.01) is image


class TestNeutralLuma:
    def test_removes_a_channel_cast(self) -> None:
        rng = np.random.default_rng(3)
        base = rng.integers(30, 220, (200, 200), dtype=np.int32)
        cast = np.stack([base * 1.3, base * 1.0, base * 0.8], axis=-1)
        grey = prep.neutral_luma(np.clip(cast, 0, 255).astype(np.uint8))
        assert grey.ndim == 2
        # Monotonic response to the original signal survives the balance.
        assert np.corrcoef(grey.ravel(), base.ravel())[0, 1] > 0.99

    def test_keeps_headroom_at_both_ends(self) -> None:
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        grey = prep.neutral_luma(image, headroom=0.05)
        assert grey.min() > 0
        assert grey.max() < 255


class TestGrade:
    def test_levels_land_on_the_requested_points(self) -> None:
        rng = np.random.default_rng(5)
        lightness = rng.integers(60, 180, (300, 300), dtype=np.uint8)
        out = grade.set_levels(lightness, black_point=4.0, white_point=248.0)
        assert out.min() == pytest.approx(4, abs=6)
        assert out.max() == pytest.approx(248, abs=6)

    def test_s_curve_is_monotonic_and_fixes_the_endpoints(self) -> None:
        ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
        curved = grade.s_curve(ramp, 0.3).ravel().astype(int)
        assert np.all(np.diff(curved) >= 0)
        assert curved[0] == 0 and curved[-1] == 255

    def test_zero_strength_changes_nothing(self) -> None:
        ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
        assert np.array_equal(grade.s_curve(ramp, 0.0), ramp)

    def test_grading_leaves_chroma_untouched(self) -> None:
        rng = np.random.default_rng(6)
        # A smooth frame, not noise: per-pixel LAB round-trip quantisation on
        # random data swamps the effect being measured.
        image = cv2.GaussianBlur(
            rng.integers(0, 256, (120, 120, 3), dtype=np.uint8), (0, 0), 6
        )
        before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(int)
        after_image, _ = grade.grade(image, RestoreConfig())
        after = cv2.cvtColor(after_image, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(int)
        # Only the LAB round trip's own quantisation should differ.
        assert np.abs(after - before).mean() < 2.0


class TestChroma:
    def test_guided_filter_preserves_a_constant_field(self) -> None:
        guide = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        source = np.full((64, 64, 2), 7.0, dtype=np.float32)
        out = colorize._guided_filter(source, guide, 8, 1e-4)
        assert np.allclose(out, 7.0, atol=1e-2)

    def test_bias_is_measured_from_the_highlights(self) -> None:
        lightness = np.zeros((100, 100), dtype=np.float32)
        lightness[:5] = 95.0  # the bright band
        chroma = np.zeros((100, 100, 2), dtype=np.float32)
        chroma[:5] = (6.0, -2.0)
        bias = colorize._chroma_bias(chroma, lightness, quantile=0.95)
        assert bias == pytest.approx([6.0, -2.0], abs=0.5)

    def test_shadow_rolloff_spares_highlights_and_damps_shadows(self) -> None:
        lightness = np.array([[0.02, 0.25, 0.5, 0.9]], dtype=np.float32)
        scale = colorize._shadow_rolloff(lightness, floor=0.4, knee=0.45)
        assert scale[0, 0] == pytest.approx(0.4, abs=1e-6)
        assert scale[0, 2] == pytest.approx(1.0, abs=1e-6)
        assert scale[0, 3] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(scale[0]) >= 0)


class TestDefects:
    def test_finds_a_speck_on_a_smooth_field(self) -> None:
        image = np.full((400, 400, 3), 90, dtype=np.uint8)
        image[200:203, 200:203] = 220
        mask = defects.detect(image, RestoreConfig())
        assert mask[199:204, 199:204].sum() >= 4

    def test_repair_removes_the_speck(self) -> None:
        image = np.full((400, 400, 3), 90, dtype=np.uint8)
        image[200:203, 200:203] = 220
        repaired, report = defects.repair(image, RestoreConfig())
        assert report["defect_pixels"] > 0
        assert int(repaired[201, 201, 0]) < 140

    def test_disabled_is_a_pass_through(self) -> None:
        image = np.full((60, 60, 3), 90, dtype=np.uint8)
        out, report = defects.repair(image, RestoreConfig(remove_defects=False))
        assert out is image and report["enabled"] is False


class TestUpscaleRamp:
    def test_ramp_rises_and_falls_within_the_overlap(self) -> None:
        weight = _ramp(100, 20)
        assert weight[0] == pytest.approx(0.0, abs=1e-6)
        assert weight[50] == pytest.approx(1.0)
        assert weight[-1] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(weight[:20]) >= -1e-6)

    def test_zero_overlap_is_flat(self) -> None:
        assert np.all(_ramp(30, 0) == 1.0)


class TestCli:
    def test_collects_images_from_a_directory(self, tmp_path) -> None:
        for name in ("b.png", "a.jpg", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")
        found = collect_inputs([str(tmp_path)])
        assert [p.name for p in found] == ["a.jpg", "b.png"]

    def test_rejects_a_non_image_argument(self, tmp_path) -> None:
        target = tmp_path / "notes.txt"
        target.write_text("x")
        with pytest.raises(SystemExit):
            collect_inputs([str(target)])
