"""CPU tests for the SpatialVID dense disparity labels on synthetic inverse depth."""

import dataclasses
import math
import zipfile
from fractions import Fraction

import numpy as np
import pytest
import torch

from minwm.processors.latent import DISP_CHANNELS, disparity_to_channels
from tests.test_spatialvid_plan import (
    KEEP_ALL,
    _annotation,
    _clip_id_with_late_start,
    _forward_trajectory,
    _meta,
)
from tools.data.wan21 import spatialvid_disparity as sd
from tools.data.wan21.spatialvid import ClipPlan, plan_clip

T, H, W = 20, 328, 584
# Depth columns whose centres lie inside the crop window [16, 1264), by hand: the centre of
# column c is (c + 0.5) / 584 · 1280, so 6 -> 14.2, 7 -> 16.4, 576 -> 1263.6, 577 -> 1265.8.
WINDOW = slice(7, 577)


def _ramp(near=0.6, far=0.1) -> np.ndarray:
    """``(20, 328, 584)`` float16 inverse depth growing from top (far) to bottom (near)."""
    rows = np.linspace(far, near, H, dtype=np.float32)[:, None]
    return np.broadcast_to(rows, (T, H, W)).astype(np.float16)


def _reference(inverse_depth: np.ndarray) -> np.ndarray:
    """Straight transcription of decisions.md, with exact index arithmetic."""
    clean = inverse_depth.astype(np.float32)
    clean[clean <= 0] = 0.005
    p90 = np.quantile(clean[:4, :, WINDOW], 0.9, method="higher")
    d = np.clip(0.9 * clean / p90, 0, 1)
    cols = [
        math.floor((16 + Fraction(3, 2) * (u + Fraction(1, 2))) / 1280 * 584) for u in range(832)
    ]
    rows = [math.floor((v + Fraction(1, 2)) / 480 * 328) for v in range(480)]
    dense = d[:, rows][:, :, cols]
    # The 90% "higher" quantile of 4 samples is the 4th smallest, i.e. the max.
    pooled = dense.reshape(T, 240, 2, 416, 2).max(axis=(2, 4))
    return pooled[:, None].astype(np.float16)


# --- index mapping -----------------------------------------------------------------------------


def test_depth_columns_and_rows_at_crop_edges():
    cols, rows = sd.depth_columns(), sd.depth_rows()
    assert cols.shape == (832,) and rows.shape == (480,)
    # u = 0: (16 + 0.75) / 1280 · 584 = 7.64; u = 1: 8.33; u = 831: 1263.25 / 1280 · 584 = 576.4.
    assert cols[:2].tolist() == [7, 8] and cols[-1] == 576
    # v = 0: 0.5 / 480 · 328 = 0.34; v = 479: 479.5 / 480 · 328 = 327.7.
    assert rows[0] == 0 and rows[-1] == 327
    exact_cols = [
        math.floor((16 + Fraction(3, 2) * (u + Fraction(1, 2))) / 1280 * 584) for u in range(832)
    ]
    exact_rows = [math.floor((v + Fraction(1, 2)) / 480 * 328) for v in range(480)]
    assert cols.tolist() == exact_cols and rows.tolist() == exact_rows


def test_crop_window_columns_are_the_sampled_columns():
    mask = sd.crop_window_columns()
    assert mask.shape == (W,)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(584)[WINDOW])
    # The output samples every in-window column and nothing outside it.
    np.testing.assert_array_equal(np.unique(sd.depth_columns()), np.flatnonzero(mask))


# --- normalization -----------------------------------------------------------------------------


def test_matches_reference():
    rng = np.random.default_rng(0)
    inv = (_ramp() * rng.uniform(0.5, 1.5, (T, H, W))).astype(np.float16)
    inv[:, :40, 100:300] = -0.01
    d, _ = sd.disparity_from_inverse_depth(inv)
    np.testing.assert_array_equal(d, _reference(inv))


def test_negative_and_zero_become_far_before_p90():
    inv = np.full((T, H, W), -0.01, np.float16)
    inv[:, :, :100] = 0
    inv[:, -20:] = 1.0  # 6% of pixels, inside the bottom 10% of rows
    d, stats = sd.disparity_from_inverse_depth(inv)
    # 94% of the pixels are sky, so P90 is the sky value itself, not −0.01 or 0.
    assert stats.p90 == pytest.approx(0.005)
    sky = d[:, 0, :100]
    assert np.all(sky == np.float16(0.9))
    assert np.all(d[:, 0, -5:] == 1)


def test_p90_uses_latent_frames_0_to_3_inside_the_crop_window():
    cols = np.arange(W, dtype=np.float32)[None, :]
    rows = np.arange(H, dtype=np.float32)[:, None]
    inv = np.broadcast_to(0.1 + 0.001 * cols + 0.002 * rows, (T, H, W)).astype(np.float16)
    inv[:, :, :7] = inv[:, :, 577:] = 50.0
    inv[4:] = (inv[4:].astype(np.float32) * 3).astype(np.float16)
    _, stats = sd.disparity_from_inverse_depth(inv)
    expected = np.quantile(inv[:4, :, WINDOW].astype(np.float32), 0.9, method="higher")
    assert stats.p90 == expected
    assert stats.p90 != np.quantile(inv[:4].astype(np.float32), 0.9, method="higher")
    assert stats.p90 != np.quantile(inv[:, :, WINDOW].astype(np.float32), 0.9, method="higher")
    assert stats.p90 in inv[:4, :, WINDOW]


def test_clips_to_unit_range():
    # Inverse depth 2x² over rows x in [0, 1]: P90 is about 2 · 0.9², so values above P90 / 0.9
    # (rows with x > 0.95) clip, and the top row (0) becomes sky.
    rows = 2 * np.linspace(0, 1, H, dtype=np.float32) ** 2
    inv = np.broadcast_to(rows[:, None], (T, H, W)).astype(np.float16)
    d, stats = sd.disparity_from_inverse_depth(inv)
    assert d.min() >= 0 and d.max() == 1
    ratio = 0.9 * np.maximum(inv.astype(np.float32), 0.005) / stats.p90
    sampled = ratio[:, sd.depth_rows()][:, :, sd.depth_columns()]
    assert stats.clipped_fraction == pytest.approx(np.mean(sampled > 1))
    assert 0.03 < stats.clipped_fraction < 0.08
    assert stats.near_below_fraction == 1.0


# --- 2×2 aggregation ---------------------------------------------------------------------------


def test_pool_quantile_picks_a_sample():
    far, near = 0.1, 0.8
    block = np.array([[far, far], [far, near]], np.float32)
    assert sd.pool_quantile(block).item() == np.float32(near)
    # Linear interpolation would give 0.47; "higher" returns a sample.
    spread = np.array([[0.2, 0.5], [0.4, 0.3]], np.float32)
    assert sd.pool_quantile(spread).item() == np.float32(0.5)


def test_pool_quantile_keeps_blocks_apart():
    d = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    out = sd.pool_quantile(d)
    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out, d.reshape(2, 2, 2, 3, 2).max(axis=(2, 4)))


# --- time --------------------------------------------------------------------------------------


def test_latent_frames_take_the_nearest_depth_frame_after_a_late_start(monkeypatch):
    fps = 30.0
    frames = np.arange(0, 301, 5)
    clip_id = _clip_id_with_late_start(frames, fps)
    ann = _annotation(_forward_trajectory(frames, fps), frames, clip_id=clip_id)
    # Shift column 0 so a mix-up of annotation index and position shows.
    ann = dataclasses.replace(ann, indices=ann.indices + 1000)
    plan = plan_clip(ann, _meta(clip_id=clip_id, fps=fps), **KEEP_ALL)
    assert plan.start_frame > 0

    # Every depth frame carries its annotation index in rows 100–199, below every value P90
    # could take, so P90 is the same whatever frames are picked.
    def tagged(index: int) -> np.ndarray:
        frame = _ramp()[0].copy()
        frame[100:200] = 0.001 * (index - 999)
        return frame

    requested = []

    def fake_read(zip_path, indices):
        requested.extend(int(i) for i in indices)
        return {int(i): tagged(int(i)) for i in indices}

    monkeypatch.setattr(sd, "read_inverse_depth", fake_read)
    d, stats = sd.clip_disparity("depths.zip", plan)

    expected = []
    for k in range(T):
        t = plan.source_frames[4 * k] / fps
        expected.append(1000 + int(np.argmin(np.abs(frames / fps - t))))
    assert requested == expected
    assert expected[0] == 1000 + plan.start_frame // 5
    # Output row 110 covers depth rows 150–151, inside the tag.
    tags = d[:, 0, 110].astype(np.float32)
    assert np.all(tags == tags[:, :1])
    decoded = np.rint(tags[:, 0] * stats.p90 / 0.9 / 0.001).astype(int) + 999
    assert decoded.tolist() == expected


def test_clip_disparity_refuses_a_skipped_plan():
    with pytest.raises(ValueError, match="skipped"):
        sd.clip_disparity("depths.zip", ClipPlan(clip_id="c", skip_reasons=("hdr",)))


# --- skips -------------------------------------------------------------------------------------


@pytest.mark.parametrize("inverted, skipped", [(10, False), (11, True)])
def test_skip_when_most_frames_are_nearer_at_the_top(inverted, skipped):
    inv = _ramp().copy()
    inv[:inverted] = inv[:inverted, ::-1]
    if skipped:
        with pytest.raises(sd.DisparitySkip) as e:
            sd.disparity_from_inverse_depth(inv)
        assert e.value.reason == sd.SKIP_INVERTED_DEPTH
    else:
        _, stats = sd.disparity_from_inverse_depth(inv)
        assert stats.near_below_fraction == 0.5


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_skip_on_nan_or_inf(bad):
    inv = _ramp().copy()
    inv[7, 200, 300] = bad
    with pytest.raises(sd.DisparitySkip, match="1 NaN/inf") as e:
        sd.disparity_from_inverse_depth(inv)
    assert e.value.reason == sd.SKIP_INVALID_DEPTH
    assert isinstance(e.value, ValueError)


# --- output ------------------------------------------------------------------------------------


def test_output_shape_dtype_and_training_layout():
    d, stats = sd.disparity_from_inverse_depth(_ramp())
    assert d.shape == sd.DISPARITY_SHAPE == (20, 1, 240, 416)
    assert d.dtype == np.float16
    assert isinstance(stats.p90, float)
    channels = disparity_to_channels(torch.from_numpy(d).float())
    assert channels.shape == (20, DISP_CHANNELS, 60, 104)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="want"):
        sd.disparity_from_inverse_depth(np.zeros((20, 480, 832), np.float16))


# --- EXR reader --------------------------------------------------------------------------------


def test_read_inverse_depth_from_zip(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    frames = {i: (_ramp()[0] + np.float16(0.01 * i)).astype(np.float16) for i in (3, 4)}
    archive = tmp_path / "depths.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i, frame in frames.items():
            path = tmp_path / f"{i:05d}.exr"
            header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
            with OpenEXR.File(header, {"Z": frame}) as exr:
                exr.write(str(path))
            z.write(path, f"{i:05d}.exr")
    out = sd.read_inverse_depth(str(archive), [4, 3, 4])
    assert sorted(out) == [3, 4]
    for i, frame in frames.items():
        assert out[i].dtype == np.float16
        np.testing.assert_array_equal(out[i], frame)
