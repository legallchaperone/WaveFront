"""Plan one SpatialVID-HQ clip for the WaveFront LMDB: frame times, poses, intrinsics, depth frames.

This is the clip-planning step of the SpatialVID -> LMDB preprocessing. It reads only a clip's
annotation directory and its metadata row, and follows the contract in
``docs/ai-generated/decisions.md`` (section "2026-11-01：SpatialVID 数据格式"). It decodes no
video, reads no depth and writes nothing.

SpatialVID ``poses.npy`` rows are w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]`` (the lietorch SE3
layout written by the DROID-SLAM / MegaSaM tracker; the official ``utils/quat_to_mat.py`` and
``utils/pack_clip_assets.py`` invert them to get c2w). ``intrinsics.npy`` rows are normalized
``[fx/W, fy/H, cx/W, cy/H]``. ``indexes.txt`` column 0 is the annotation index (also the member
name in the depth zip), column 1 the source frame number.

Usage::

    python tools/data/wan21/spatialvid.py \\
        --metadata /path/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv \\
        --annotations_root /path/SpatialVID-HQ/extracted/annotations/SpatialVID \\
        --group 1 --num_clips 5 --min_move_dist 0.1 --max_dynamic_ratio 0.3
"""

import argparse
import csv
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

OUTPUT_FPS = 16
NUM_PIXEL_FRAMES = 45
PIXEL_FRAMES_PER_LATENT = 4
NUM_LATENT_FRAMES = (NUM_PIXEL_FRAMES - 1) // PIXEL_FRAMES_PER_LATENT + 1
MIN_SPAN_SECONDS = 2.75
SOURCE_WIDTH, SOURCE_HEIGHT = 1280, 720
CROP_WIDTH = 1248
MAX_CROPPED_FX = 1.5
TRANSLATION_SCALE = 1.54

SKIP_SHORT_SPAN = "short_span"
SKIP_LONG_FOCAL = "long_focal"
SKIP_NEAR_STATIC = "near_static"
SKIP_TOO_DYNAMIC = "too_dynamic"


@dataclass(frozen=True)
class ClipAnnotation:
    """The parts of a SpatialVID annotation directory the planner uses.

    Attributes:
        clip_id (str): clip id (the annotation directory name).
        indices (np.ndarray): ``(N,)`` int64 annotation index of each annotation frame
            (``indexes.txt`` column 0; the member name in the depth zip).
        frames (np.ndarray): ``(N,)`` int64 source frame number of each annotation frame
            (``indexes.txt`` column 1).
        poses (np.ndarray): ``(N, 7)`` float64 w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]``.
        intrinsics (np.ndarray): ``(N, 4)`` float64 ``[fx/W, fy/H, cx/W, cy/H]``.
    """

    clip_id: str
    indices: np.ndarray
    frames: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray


@dataclass(frozen=True)
class ClipPlan:
    """What to extract from one clip. Array fields are ``None`` when the clip is skipped.

    Attributes:
        clip_id (str): clip id.
        skip_reasons (tuple[str, ...]): every ``SKIP_*`` reason that applies; empty if kept.
        source_frames (np.ndarray | None): ``(45,)`` int64 source frame of each output pixel
            frame.
        source_times (np.ndarray | None): ``(45,)`` float64 time in seconds of those frames.
        poses (np.ndarray | None): ``(12, 7)`` float32 w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]``
            per latent frame, aligned to latent frame 0 and scaled by ``TRANSLATION_SCALE``.
        intrinsics (np.ndarray | None): ``(4,)`` float32 ``[fx/W, fy/H, cx/W, cy/H]`` after the
            crop to 1248×720 and resize to 832×480.
        depth_indices (np.ndarray | None): ``(12,)`` int64 annotation index of the depth frame
            for each latent frame.
    """

    clip_id: str
    skip_reasons: tuple[str, ...]
    source_frames: np.ndarray | None = None
    source_times: np.ndarray | None = None
    poses: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    depth_indices: np.ndarray | None = None

    @property
    def skipped(self) -> bool:
        return bool(self.skip_reasons)


def load_annotation(ann_dir: str) -> ClipAnnotation:
    """Read ``indexes.txt``, ``poses.npy`` and ``intrinsics.npy`` of one clip.

    Args:
        ann_dir (str): the clip's annotation directory.

    Returns:
        ClipAnnotation: the clip's annotation frames, poses and intrinsics.

    Raises:
        ValueError: if the three files disagree on the number of annotation frames.
    """
    indexes = np.loadtxt(os.path.join(ann_dir, "indexes.txt"), dtype=np.int64, ndmin=2)
    poses = np.load(os.path.join(ann_dir, "poses.npy")).astype(np.float64)
    intrinsics = np.load(os.path.join(ann_dir, "intrinsics.npy")).astype(np.float64)
    if not (len(indexes) == len(poses) == len(intrinsics)):
        raise ValueError(
            f"{ann_dir}: {len(indexes)} indexes, {len(poses)} poses, "
            f"{len(intrinsics)} intrinsics"
        )
    return ClipAnnotation(
        clip_id=os.path.basename(os.path.normpath(ann_dir)),
        indices=indexes[:, 0],
        frames=indexes[:, 1],
        poses=poses,
        intrinsics=intrinsics,
    )


def output_source_frames(fps: float, num_frames: int = NUM_PIXEL_FRAMES) -> np.ndarray:
    """Source frame of output pixel frame ``j``: ``round(j·fps/16)``, halves rounded up.

    Args:
        fps (float): source frame rate.
        num_frames (int): number of output pixel frames.

    Returns:
        np.ndarray: ``(num_frames,)`` int64 source frame numbers.
    """
    # Half-up, not numpy's half-to-even: at 24 fps j·1.5 hits .5 on every odd j, and
    # half-to-even would make the frame spacing irregular (2,1,1,2,...) instead of (2,1,2,1,...).
    return np.floor(np.arange(num_frames) * fps / OUTPUT_FPS + 0.5).astype(np.int64)


def crop_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """Map normalized intrinsics of a 1280×720 frame through the centre crop to 1248×720.

    The later 2/3 resize to 832×480 leaves normalized intrinsics unchanged.

    Args:
        intrinsics (np.ndarray): ``(4,)`` ``[fx/W, fy/H, cx/W, cy/H]`` of the source frame.

    Returns:
        np.ndarray: ``(4,)`` float32 ``[fx/W, fy/H, cx/W, cy/H]`` of the output frame.
    """
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    margin = (SOURCE_WIDTH - CROP_WIDTH) / 2
    return np.array(
        [fx * SOURCE_WIDTH / CROP_WIDTH, fy, (cx * SOURCE_WIDTH - margin) / CROP_WIDTH, cy],
        dtype=np.float32,
    )


def pose_vec_to_matrix(poses: np.ndarray) -> np.ndarray:
    """Convert ``(T, 7)`` ``[tx, ty, tz, qx, qy, qz, qw]`` rows to ``(T, 4, 4)`` matrices.

    Args:
        poses (np.ndarray): ``(T, 7)`` translation plus scalar-last quaternion.

    Returns:
        np.ndarray: ``(T, 4, 4)`` float64 rigid transforms.
    """
    from scipy.spatial.transform import Rotation

    poses = np.asarray(poses, dtype=np.float64)
    mats = np.tile(np.eye(4), (len(poses), 1, 1))
    mats[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    mats[:, :3, 3] = poses[:, :3]
    return mats


def matrix_to_pose_vec(mats: np.ndarray) -> np.ndarray:
    """Convert ``(T, 4, 4)`` rigid transforms to ``(T, 7)`` ``[tx, ty, tz, qx, qy, qz, qw]``.

    Args:
        mats (np.ndarray): ``(T, 4, 4)`` rigid transforms.

    Returns:
        np.ndarray: ``(T, 7)`` float64 translation plus unit scalar-last quaternion.
    """
    from scipy.spatial.transform import Rotation

    quats = Rotation.from_matrix(mats[:, :3, :3]).as_quat()
    return np.concatenate([mats[:, :3, 3], quats], axis=1)


def invert_rigid(mats: np.ndarray) -> np.ndarray:
    """Invert ``(..., 4, 4)`` rigid transforms (w2c <-> c2w) as ``[Rᵀ | −Rᵀt]``."""
    rot_t = np.swapaxes(mats[..., :3, :3], -1, -2)
    out = np.zeros_like(mats)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -(rot_t @ mats[..., :3, 3:])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def interpolate_c2w(c2w: np.ndarray, frames: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate c2w poses known at ``frames`` onto ``targets``.

    The camera centre is interpolated linearly and the rotation by slerp, both on the same
    (time-proportional) frame axis. Doing it on c2w rather than w2c makes the result independent
    of the world frame, so aligning before or after interpolation gives the same poses.

    Args:
        c2w (np.ndarray): ``(N, 4, 4)`` c2w poses at the annotation frames.
        frames (np.ndarray): ``(N,)`` strictly increasing source frame numbers of those poses.
        targets (np.ndarray): ``(T,)`` source frame numbers (may be fractional) to evaluate.

    Returns:
        np.ndarray: ``(T, 4, 4)`` float64 interpolated c2w poses.

    Raises:
        ValueError: if a target lies outside ``[frames[0], frames[-1]]`` (no extrapolation).
    """
    from scipy.spatial.transform import Rotation, Slerp

    frames = np.asarray(frames, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.min() < frames[0] or targets.max() > frames[-1]:
        raise ValueError(
            f"targets [{targets.min()}, {targets.max()}] outside annotated "
            f"[{frames[0]}, {frames[-1]}]"
        )
    out = np.tile(np.eye(4), (len(targets), 1, 1))
    out[:, :3, :3] = Slerp(frames, Rotation.from_matrix(c2w[:, :3, :3]))(targets).as_matrix()
    for axis in range(3):
        out[:, axis, 3] = np.interp(targets, frames, c2w[:, axis, 3])
    return out


def latent_frame_poses(annotation: ClipAnnotation, fps: float) -> np.ndarray:
    """Poses of the latent frames, ready for the LMDB ``poses`` key.

    Latent frame ``k`` takes output pixel frame ``4k`` at its actual source time, interpolated
    between the neighbouring annotation frames, then the trajectory is aligned so latent frame 0
    is the identity and its translations are scaled by ``TRANSLATION_SCALE``.

    Args:
        annotation (ClipAnnotation): the clip's annotation (w2c poses).
        fps (float): source frame rate from the metadata.

    Returns:
        np.ndarray: ``(12, 7)`` float32 w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]``.
    """
    targets = output_source_frames(fps)[::PIXEL_FRAMES_PER_LATENT]
    c2w = interpolate_c2w(
        invert_rigid(pose_vec_to_matrix(annotation.poses)), annotation.frames, targets
    )
    aligned = invert_rigid(c2w[0]) @ c2w
    aligned[:, :3, 3] *= TRANSLATION_SCALE
    return matrix_to_pose_vec(invert_rigid(aligned)).astype(np.float32)


def nearest_annotation(frames: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Position in ``frames`` of the annotation frame nearest each target; ties go earlier.

    Args:
        frames (np.ndarray): ``(N,)`` increasing source frame numbers of the annotation frames.
        targets (np.ndarray): ``(T,)`` source frame numbers.

    Returns:
        np.ndarray: ``(T,)`` int64 positions into ``frames``.
    """
    dist = np.abs(np.asarray(targets)[:, None] - np.asarray(frames)[None, :])
    return np.argmin(dist, axis=1).astype(np.int64)


def _check_annotation(annotation: ClipAnnotation, meta: Mapping) -> None:
    clip = annotation.clip_id
    if str(meta["id"]) != clip:
        raise ValueError(f"metadata row {meta['id']} does not belong to clip {clip}")
    if str(meta["resolution"]) != f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}":
        raise ValueError(f"{clip}: resolution {meta['resolution']}, crop assumes 1280x720")
    if annotation.frames[0] != 0:
        raise ValueError(f"{clip}: first annotation frame is {annotation.frames[0]}, not 0")
    if np.any(np.diff(annotation.frames) <= 0):
        raise ValueError(f"{clip}: annotation frames are not strictly increasing")
    if not np.all(annotation.intrinsics == annotation.intrinsics[0]):
        raise ValueError(f"{clip}: intrinsics vary across annotation frames")


def plan_clip(
    annotation: ClipAnnotation,
    meta: Mapping,
    *,
    min_move_dist: float,
    max_dynamic_ratio: float,
) -> ClipPlan:
    """Plan one clip under the SpatialVID data contract.

    Args:
        annotation (ClipAnnotation): the clip's annotation.
        meta (Mapping): the clip's metadata row; uses ``id``, ``fps``, ``resolution``,
            ``moveDist`` and ``dynamicRatio``.
        min_move_dist (float): skip the clip if its ``moveDist`` is below this.
        max_dynamic_ratio (float): skip the clip if its ``dynamicRatio`` is above this.

    Returns:
        ClipPlan: the plan, or the skip reasons with no arrays.

    Raises:
        ValueError: if the metadata row is for another clip, the source is not 1280×720, the
            first annotation frame is not source frame 0, annotation frames are not increasing,
            or the intrinsics vary across annotation frames.
    """
    _check_annotation(annotation, meta)
    fps = float(meta["fps"])
    intrinsics = crop_intrinsics(annotation.intrinsics[0])

    reasons = []
    if annotation.frames[-1] / fps < MIN_SPAN_SECONDS:
        reasons.append(SKIP_SHORT_SPAN)
    if intrinsics[0] > MAX_CROPPED_FX:
        reasons.append(SKIP_LONG_FOCAL)
    if float(meta["moveDist"]) < min_move_dist:
        reasons.append(SKIP_NEAR_STATIC)
    if float(meta["dynamicRatio"]) > max_dynamic_ratio:
        reasons.append(SKIP_TOO_DYNAMIC)
    if reasons:
        return ClipPlan(clip_id=annotation.clip_id, skip_reasons=tuple(reasons))

    source_frames = output_source_frames(fps)
    latent_frames = source_frames[::PIXEL_FRAMES_PER_LATENT]
    nearest = nearest_annotation(annotation.frames, latent_frames)
    return ClipPlan(
        clip_id=annotation.clip_id,
        skip_reasons=(),
        source_frames=source_frames,
        source_times=source_frames / fps,
        poses=latent_frame_poses(annotation, fps),
        intrinsics=intrinsics,
        depth_indices=annotation.indices[nearest],
    )


def _summarize(plan: ClipPlan, meta: Mapping) -> str:
    head = f"{plan.clip_id} fps={float(meta['fps']):.3f} moveDist={float(meta['moveDist']):.3f}"
    if plan.skipped:
        return f"{head} SKIP {','.join(plan.skip_reasons)}"
    centres = invert_rigid(pose_vec_to_matrix(plan.poses))[:, :3, 3] / TRANSLATION_SCALE
    path = np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()
    angle = np.degrees(2 * np.arccos(np.clip(abs(plan.poses[-1, 6]), 0, 1)))
    return (
        f"{head} intrinsics={np.round(plan.intrinsics, 4).tolist()} "
        f"path_m={path:.3f} end_centre_m={np.round(centres[-1], 3).tolist()} "
        f"end_rot_deg={angle:.1f} depth_indices={plan.depth_indices.tolist()}"
    )


def main() -> None:
    """Plan a few clips of one group and print a one-line summary per clip."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", required=True, help="SpatialVID_HQ_metadata.csv")
    parser.add_argument(
        "--annotations_root",
        required=True,
        help="directory that the metadata 'annotation path' column is relative to",
    )
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--num_clips", type=int, default=5)
    parser.add_argument("--min_move_dist", type=float, required=True)
    parser.add_argument("--max_dynamic_ratio", type=float, required=True)
    args = parser.parse_args()

    with open(args.metadata, newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["group id"]) == args.group]
    for row in sorted(rows, key=lambda r: r["id"])[: args.num_clips]:
        annotation = load_annotation(os.path.join(args.annotations_root, row["annotation path"]))
        plan = plan_clip(
            annotation,
            row,
            min_move_dist=args.min_move_dist,
            max_dynamic_ratio=args.max_dynamic_ratio,
        )
        print(_summarize(plan, row))


if __name__ == "__main__":
    main()
