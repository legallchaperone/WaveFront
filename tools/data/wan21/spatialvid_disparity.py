"""Dense disparity labels of one planned SpatialVID-HQ clip, for the LMDB ``disparity`` key.

Follows ``docs/ai-generated/decisions.md``, section "2026-11-01：SpatialVID 数据格式", bullets
**深度图** and **disparity** 1–5. A clip's depth zip holds one ``{i:05d}.exr`` per annotation
frame ``i`` (``indexes.txt`` column 0), 328×584, whose single ``Z`` channel is inverse depth
(float16). Latent frame ``k`` takes the depth frame the planner assigned it
(:attr:`~tools.data.wan21.spatialvid.ClipPlan.depth_indices`, nearest in time to pixel frame
``4k``). Per clip:

1. Negative and zero inverse depth (both encode sky) become ``SKY_INVERSE_DEPTH``.
2. ``P90`` is the 90% quantile over the depth frames of latent frames 0–3, using the depth
   pixels whose centres fall inside the 1248×720 crop window.
3. ``d = clip(0.9 · (1/z) / P90, 0, 1)``, with ``γ = 1``.
4. ``d`` is sampled to 480×832 with the same crop as the RGB frames (nearest neighbour), then
   each 2×2 block is reduced to its 90% quantile, giving 240×416.

Only :func:`read_inverse_depth` touches EXR files; the rest is numpy on arrays.
"""

import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tools.data.wan21.spatialvid import (
    CROP_WIDTH,
    NUM_LATENT_FRAMES,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SOURCE_WIDTH,
    ClipPlan,
)

DEPTH_HEIGHT, DEPTH_WIDTH = 328, 584
DISPARITY_POOL = 2
DISPARITY_SHAPE = (
    NUM_LATENT_FRAMES,
    1,
    OUTPUT_HEIGHT // DISPARITY_POOL,
    OUTPUT_WIDTH // DISPARITY_POOL,
)
SKY_INVERSE_DEPTH = 0.005
QUANTILE = 0.9
P90_LATENT_FRAMES = 4
NORMALIZED_P90 = 0.9
EDGE_ROWS_FRACTION = 0.1
MIN_NEAR_BELOW_FRACTION = 0.5

SKIP_INVALID_DEPTH = "invalid_depth"
SKIP_INVERTED_DEPTH = "inverted_depth"


class DisparitySkip(ValueError):
    """The clip's depth breaks a contract check, so the clip gets no disparity label.

    Args:
        reason (str): ``SKIP_INVALID_DEPTH`` or ``SKIP_INVERTED_DEPTH``.
        message (str): what was measured.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass(frozen=True)
class DisparityStats:
    """Per-clip numbers for the preprocessing log.

    Attributes:
        p90 (float): the clip's ``P90`` of inverse depth, after the sky replacement.
        clipped_fraction (float): fraction of the 480×832 samples of all latent frames whose
            ``0.9 · (1/z) / P90`` exceeds 1 before clipping.
        near_below_fraction (float): fraction of latent frames whose bottom 10% of rows has a
            higher median inverse depth than the top 10%.
    """

    p90: float
    clipped_fraction: float
    near_below_fraction: float


def depth_columns() -> np.ndarray:
    """Depth-map column of each output column ``u``: ``⌊(16 + 1.5(u + 0.5)) / 1280 · 584⌋``.

    Returns:
        np.ndarray: ``(832,)`` int64 column indices.
    """
    margin = (SOURCE_WIDTH - CROP_WIDTH) / 2
    source_x = margin + CROP_WIDTH / OUTPUT_WIDTH * (np.arange(OUTPUT_WIDTH) + 0.5)
    return np.floor(source_x / SOURCE_WIDTH * DEPTH_WIDTH).astype(np.int64)


def depth_rows() -> np.ndarray:
    """Depth-map row of each output row ``v``: ``⌊(v + 0.5) / 480 · 328⌋``.

    Returns:
        np.ndarray: ``(480,)`` int64 row indices.
    """
    centres = np.arange(OUTPUT_HEIGHT) + 0.5
    return np.floor(centres / OUTPUT_HEIGHT * DEPTH_HEIGHT).astype(np.int64)


def crop_window_columns() -> np.ndarray:
    """Depth-map columns whose centres fall inside the 1248-pixel-wide crop window.

    The crop keeps every row, so these columns are the "cropped native resolution" that
    ``P90`` is computed on.

    Returns:
        np.ndarray: ``(584,)`` bool mask over depth-map columns.
    """
    margin = (SOURCE_WIDTH - CROP_WIDTH) / 2
    centres = (np.arange(DEPTH_WIDTH) + 0.5) / DEPTH_WIDTH * SOURCE_WIDTH
    return (centres >= margin) & (centres < margin + CROP_WIDTH)


def replace_sky(inverse_depth: np.ndarray) -> np.ndarray:
    """Replace negative and zero inverse depth, the two sky encodings, by ``SKY_INVERSE_DEPTH``.

    Args:
        inverse_depth (np.ndarray): inverse depth of any shape; NaN and inf pass through.

    Returns:
        np.ndarray: float32 copy with the replacement applied.
    """
    out = np.asarray(inverse_depth, dtype=np.float32).copy()
    out[out <= 0] = SKY_INVERSE_DEPTH
    return out


def near_below_fraction(inverse_depth: np.ndarray) -> float:
    """Fraction of frames whose bottom rows are nearer than their top rows.

    Compares the median inverse depth of the bottom and top ``EDGE_ROWS_FRACTION`` of rows,
    rounded to whole rows.

    Args:
        inverse_depth (np.ndarray): ``(T, H, W)`` inverse depth.

    Returns:
        float: fraction of the ``T`` frames with a higher bottom median.
    """
    rows = max(1, round(EDGE_ROWS_FRACTION * inverse_depth.shape[1]))
    top = np.median(inverse_depth[:, :rows].reshape(len(inverse_depth), -1), axis=1)
    bottom = np.median(inverse_depth[:, -rows:].reshape(len(inverse_depth), -1), axis=1)
    return float(np.mean(bottom > top))


def pool_quantile(d: np.ndarray) -> np.ndarray:
    """Reduce each ``DISPARITY_POOL``² block of the last two axes to its 90% quantile.

    Uses ``method="higher"``, so every output is one of its block's samples.

    Args:
        d (np.ndarray): ``(..., H, W)`` with ``H`` and ``W`` divisible by ``DISPARITY_POOL``.

    Returns:
        np.ndarray: ``(..., H / DISPARITY_POOL, W / DISPARITY_POOL)``.
    """
    p = DISPARITY_POOL
    *lead, h, w = d.shape
    blocks = d.reshape(*lead, h // p, p, w // p, p).swapaxes(-3, -2)
    return np.quantile(
        blocks.reshape(*lead, h // p, w // p, p * p), QUANTILE, axis=-1, method="higher"
    )


def disparity_from_inverse_depth(
    inverse_depth: np.ndarray,
) -> tuple[np.ndarray, DisparityStats]:
    """Turn the depth frames of a clip's latent frames into the LMDB ``disparity`` value.

    Args:
        inverse_depth (np.ndarray): ``(20, 328, 584)`` inverse depth as stored in the EXR
            ``Z`` channel; row ``k`` is the depth frame of latent frame ``k``.

    Returns:
        tuple[np.ndarray, DisparityStats]: ``(20, 1, 240, 416)`` float16 ``d`` in ``[0, 1]``
        (1 nearest), and the clip's log numbers.

    Raises:
        ValueError: if ``inverse_depth`` has the wrong shape.
        DisparitySkip: ``SKIP_INVALID_DEPTH`` if any value is NaN or inf;
            ``SKIP_INVERTED_DEPTH`` if fewer than ``MIN_NEAR_BELOW_FRACTION`` of the frames
            are nearer at the bottom than at the top.
    """
    expected = (NUM_LATENT_FRAMES, DEPTH_HEIGHT, DEPTH_WIDTH)
    if inverse_depth.shape != expected:
        raise ValueError(f"inverse depth is {inverse_depth.shape}, want {expected}")
    invalid = int(np.count_nonzero(~np.isfinite(inverse_depth)))
    if invalid:
        raise DisparitySkip(SKIP_INVALID_DEPTH, f"{invalid} NaN/inf depth pixels")

    clean = replace_sky(inverse_depth)
    cropped = clean[:, :, crop_window_columns()]
    below = near_below_fraction(cropped)
    if below < MIN_NEAR_BELOW_FRACTION:
        raise DisparitySkip(
            SKIP_INVERTED_DEPTH, f"only {below:.2f} of frames are nearer at the bottom"
        )
    p90 = float(np.quantile(cropped[:P90_LATENT_FRAMES], QUANTILE, method="higher"))

    sampled = clean[:, depth_rows()[:, None], depth_columns()[None, :]]
    ratio = NORMALIZED_P90 * sampled / p90
    d = pool_quantile(np.clip(ratio, 0.0, 1.0))
    stats = DisparityStats(
        p90=p90,
        clipped_fraction=float(np.mean(ratio > 1.0)),
        near_below_fraction=below,
    )
    return d[:, None].astype(np.float16), stats


def read_inverse_depth(zip_path: str, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Read the ``Z`` channel of depth frames ``{i:05d}.exr`` from a clip's depth zip.

    Members are extracted to a temporary directory because ``OpenEXR.File`` reads from a path.

    Args:
        zip_path (str): the clip's depth zip.
        indices (Sequence[int]): annotation indices to read; repeats are read once.

    Returns:
        dict[int, np.ndarray]: ``(328, 584)`` inverse depth per annotation index, in the
        EXR's stored dtype (float16).

    Raises:
        KeyError: if a member is missing from the zip.
        ValueError: if a depth frame is not 328×584.
    """
    import OpenEXR

    out = {}
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp:
        for index in sorted(set(int(i) for i in indices)):
            path = archive.extract(f"{index:05d}.exr", tmp)
            with OpenEXR.File(path) as exr:
                z = np.asarray(exr.channels()["Z"].pixels)
            if z.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
                raise ValueError(f"{zip_path}: {index:05d}.exr is {z.shape}")
            out[index] = z
    return out


def clip_disparity(zip_path: str, plan: ClipPlan) -> tuple[np.ndarray, DisparityStats]:
    """The LMDB ``disparity`` value of a kept clip, from its depth zip.

    Args:
        zip_path (str): the clip's depth zip.
        plan (ClipPlan): the clip's plan from :func:`~tools.data.wan21.spatialvid.plan_clip`.

    Returns:
        tuple[np.ndarray, DisparityStats]: as :func:`disparity_from_inverse_depth`.

    Raises:
        ValueError: if the plan is a skip.
        DisparitySkip: as :func:`disparity_from_inverse_depth`.
    """
    if plan.skipped:
        raise ValueError(f"{plan.clip_id}: skipped by the planner {plan.skip_reasons}")
    frames = read_inverse_depth(zip_path, plan.depth_indices)
    return disparity_from_inverse_depth(np.stack([frames[int(i)] for i in plan.depth_indices]))
