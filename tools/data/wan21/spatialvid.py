"""Plan one SpatialVID-HQ clip for the WaveFront LMDB: window, frames, poses, intrinsics, depth.

This is the clip-planning step of the SpatialVID -> LMDB preprocessing, plus the per-clip readers
the LMDB writer (``build_spatialvid_lmdb.py``) uses: the planned frames and the caption. Planning
reads a clip's annotation directory, its metadata row and its video stream header, and follows
the contract in ``docs/ai-generated/decisions.md`` (section "2026-11-01：SpatialVID 数据格式"). The
CLI selects clips; it decodes no video and reads no depth.

Each clip yields one 77-pixel-frame window starting at an annotation frame, chosen from the clip
id so every run picks the same one.

SpatialVID ``poses.npy`` rows are w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]`` (the lietorch SE3
layout written by the DROID-SLAM / MegaSaM tracker; the official ``utils/quat_to_mat.py`` and
``utils/pack_clip_assets.py`` invert them to get c2w). ``intrinsics.npy`` rows are normalized
``[fx/W, fy/H, cx/W, cy/H]``. ``indexes.txt`` column 0 is the annotation index (also the member
name in the depth zip), column 1 the source frame number.

Usage::

    python tools/data/wan21/spatialvid.py \\
        --metadata /path/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv \\
        --annotations_root /path/SpatialVID-HQ/extracted/annotations/SpatialVID \\
        --videos_root /path/SpatialVID-HQ/extracted/videos/SpatialVID \\
        --group 1 --output clip_ids_group_0001.txt
"""

import argparse
import csv
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

OUTPUT_FPS = 16
NUM_PIXEL_FRAMES = 77
PIXEL_FRAMES_PER_LATENT = 4
NUM_LATENT_FRAMES = (NUM_PIXEL_FRAMES - 1) // PIXEL_FRAMES_PER_LATENT + 1
WINDOW_SECONDS = (NUM_PIXEL_FRAMES - 1) / OUTPUT_FPS
SOURCE_WIDTH, SOURCE_HEIGHT = 1280, 720
CROP_WIDTH = 1248
OUTPUT_WIDTH, OUTPUT_HEIGHT = 832, 480
MAX_CROPPED_FX = 1.5
TRANSLATION_SCALE = 1.54
# CLI defaults only; plan_clip takes both thresholds explicitly.
MIN_WINDOW_PATH = 0.05
MAX_DYNAMIC_RATIO = 0.3

SKIP_SHORT_SPAN = "short_span"
SKIP_HDR = "hdr"
SKIP_LONG_FOCAL = "long_focal"
SKIP_NEAR_STATIC = "near_static"
SKIP_TOO_DYNAMIC = "too_dynamic"

# HLG (arib-std-b67) and PQ (smpte2084), as FFmpeg numbers them in AVColorTransferCharacteristic.
HDR_TRANSFERS = (16, 18)

CAPTION_FIELDS = ("SceneSummary", "SceneDescription", "ShotImmersion", "CameraMotion")
DEFAULT_CAPTION_FIELD = "SceneDescription"


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
        start_frame (int | None): source frame of the window's first pixel frame, an annotation
            frame; ``None`` if the clip has no room for a window.
        window_path (float | None): camera-centre path over the latent frames of the window in
            metres (before ``TRANSLATION_SCALE``); ``None`` if the clip has no room for a window.
        source_frames (np.ndarray | None): ``(77,)`` int64 source frame of each output pixel
            frame.
        source_times (np.ndarray | None): ``(77,)`` float64 time in seconds of those frames.
        poses (np.ndarray | None): ``(20, 7)`` float32 w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]``
            per latent frame, aligned to latent frame 0 and scaled by ``TRANSLATION_SCALE``.
        intrinsics (np.ndarray | None): ``(4,)`` float32 ``[fx/W, fy/H, cx/W, cy/H]`` after the
            crop to 1248×720 and resize to 832×480.
        depth_indices (np.ndarray | None): ``(20,)`` int64 annotation index of the depth frame
            for each latent frame.
    """

    clip_id: str
    skip_reasons: tuple[str, ...]
    start_frame: int | None = None
    window_path: float | None = None
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


def output_source_frames(
    fps: float, start_frame: int = 0, num_frames: int = NUM_PIXEL_FRAMES
) -> np.ndarray:
    """Source frame of output pixel frame ``j``: ``round(t0·fps + j·fps/16)``, halves rounded up.

    Args:
        fps (float): source frame rate.
        start_frame (int): source frame of the window start, i.e. ``t0·fps``.
        num_frames (int): number of output pixel frames.

    Returns:
        np.ndarray: ``(num_frames,)`` int64 source frame numbers.
    """
    # Half-up, not numpy's half-to-even: at 24 fps j·1.5 hits .5 on every odd j, and
    # half-to-even would make the frame spacing irregular (2,1,1,2,...) instead of (2,1,2,1,...).
    offsets = np.floor(np.arange(num_frames) * fps / OUTPUT_FPS + 0.5).astype(np.int64)
    return start_frame + offsets


def window_starts(frames: np.ndarray, fps: float) -> np.ndarray:
    """Positions of the annotation frames that leave at least ``WINDOW_SECONDS`` of annotation.

    A window starting at such a frame ends at or before the last annotation frame, so its poses
    never need extrapolation.

    Args:
        frames (np.ndarray): ``(N,)`` increasing source frame numbers of the annotation frames.
        fps (float): source frame rate.

    Returns:
        np.ndarray: ``(M,)`` int64 positions into ``frames``; empty if the clip is too short.
    """
    frames = np.asarray(frames)
    return np.flatnonzero((frames[-1] - frames) / fps >= WINDOW_SECONDS).astype(np.int64)


def choose_start(clip_id: str, num_starts: int) -> int:
    """Pick one of ``num_starts`` window starts from the clip id, the same in every run.

    Uses SHA-1 of the id, not ``hash()``, whose string hashing is salted per process.

    Args:
        clip_id (str): the clip id.
        num_starts (int): number of valid starts.

    Returns:
        int: an index in ``[0, num_starts)``.
    """
    return int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % num_starts


def video_is_hdr(video_path: str) -> bool:
    """True if the video stream is HDR: more than 8 bits per sample, or an HLG / PQ transfer.

    Reads only the container and stream headers.

    Args:
        video_path (str): the clip's mp4.

    Returns:
        bool: whether the clip is HDR.

    Raises:
        ValueError: if the header has no pixel format.
    """
    import av

    with av.open(video_path) as container:
        codec = container.streams.video[0].codec_context
        if codec.format is None:
            raise ValueError(f"{video_path}: no pixel format in the stream header")
        bits = max(c.bits for c in codec.format.components)
        return bits > 8 or codec.color_trc in HDR_TRANSFERS


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


def latent_frame_c2w(annotation: ClipAnnotation, source_frames: np.ndarray) -> np.ndarray:
    """c2w poses of the latent frames in metres, aligned so latent frame 0 is the identity.

    Latent frame ``k`` takes output pixel frame ``4k`` at its actual source time, interpolated
    between the neighbouring annotation frames.

    Args:
        annotation (ClipAnnotation): the clip's annotation (w2c poses).
        source_frames (np.ndarray): ``(77,)`` source frame of each output pixel frame.

    Returns:
        np.ndarray: ``(20, 4, 4)`` float64 aligned c2w poses, translations in metres.
    """
    targets = np.asarray(source_frames)[::PIXEL_FRAMES_PER_LATENT]
    c2w = interpolate_c2w(
        invert_rigid(pose_vec_to_matrix(annotation.poses)), annotation.frames, targets
    )
    return invert_rigid(c2w[0]) @ c2w


def window_path(c2w: np.ndarray) -> float:
    """Path length of the camera centres of ``(T, 4, 4)`` c2w poses, in their units."""
    return float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum())


def _lmdb_poses(c2w: np.ndarray) -> np.ndarray:
    scaled = c2w.copy()
    scaled[:, :3, 3] *= TRANSLATION_SCALE
    return matrix_to_pose_vec(invert_rigid(scaled)).astype(np.float32)


def latent_frame_poses(annotation: ClipAnnotation, source_frames: np.ndarray) -> np.ndarray:
    """Poses of the latent frames, ready for the LMDB ``poses`` key.

    The aligned poses of :func:`latent_frame_c2w` with translations scaled by
    ``TRANSLATION_SCALE``, as w2c.

    Args:
        annotation (ClipAnnotation): the clip's annotation (w2c poses).
        source_frames (np.ndarray): ``(77,)`` source frame of each output pixel frame.

    Returns:
        np.ndarray: ``(20, 7)`` float32 w2c OpenCV ``[tx, ty, tz, qx, qy, qz, qw]``.
    """
    return _lmdb_poses(latent_frame_c2w(annotation, source_frames))


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
    hdr: bool,
    min_window_path: float,
    max_dynamic_ratio: float,
) -> ClipPlan:
    """Plan one clip under the SpatialVID data contract.

    Args:
        annotation (ClipAnnotation): the clip's annotation.
        meta (Mapping): the clip's metadata row; uses ``id``, ``fps``, ``resolution`` and
            ``dynamicRatio``.
        hdr (bool): whether the clip's video is HDR (:func:`video_is_hdr`); HDR clips are
            skipped.
        min_window_path (float): skip the clip if the camera-centre path over the latent
            frames of its window, in metres before ``TRANSLATION_SCALE``, is below this. Not
            evaluated for a clip too short to have a window.
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
    start = c2w = path = source_frames = None
    starts = window_starts(annotation.frames, fps)
    if len(starts) == 0:
        reasons.append(SKIP_SHORT_SPAN)
    else:
        start = int(annotation.frames[starts[choose_start(annotation.clip_id, len(starts))]])
        source_frames = output_source_frames(fps, start)
        c2w = latent_frame_c2w(annotation, source_frames)
        path = window_path(c2w)
    if hdr:
        reasons.append(SKIP_HDR)
    if intrinsics[0] > MAX_CROPPED_FX:
        reasons.append(SKIP_LONG_FOCAL)
    if path is not None and path < min_window_path:
        reasons.append(SKIP_NEAR_STATIC)
    if float(meta["dynamicRatio"]) > max_dynamic_ratio:
        reasons.append(SKIP_TOO_DYNAMIC)
    if reasons:
        return ClipPlan(
            clip_id=annotation.clip_id,
            skip_reasons=tuple(reasons),
            start_frame=start,
            window_path=path,
        )

    nearest = nearest_annotation(annotation.frames, source_frames[::PIXEL_FRAMES_PER_LATENT])
    return ClipPlan(
        clip_id=annotation.clip_id,
        skip_reasons=(),
        start_frame=start,
        window_path=path,
        source_frames=source_frames,
        source_times=source_frames / fps,
        poses=_lmdb_poses(c2w),
        intrinsics=intrinsics,
        depth_indices=annotation.indices[nearest],
    )


def read_metadata(
    path: str, *, group: int | None = None, clip_ids: set[str] | None = None
) -> dict[str, dict[str, str]]:
    """Read the rows of ``SpatialVID_HQ_metadata.csv`` for one group or a set of clip ids.

    Args:
        path (str): the metadata CSV.
        group (int | None): keep only rows of this ``group id``.
        clip_ids (set[str] | None): keep only rows with these ids.

    Returns:
        dict[str, dict[str, str]]: rows keyed by clip id, values as strings.
    """
    with open(path, newline="") as f:
        return {
            row["id"]: row
            for row in csv.DictReader(f)
            if (group is None or int(row["group id"]) == group)
            and (clip_ids is None or row["id"] in clip_ids)
        }


def crop_resize_frame(frame: np.ndarray) -> np.ndarray:
    """Centre-crop a 720×1280 frame to 720×1248, then area-resize it to 480×832.

    Args:
        frame (np.ndarray): ``(720, 1280, C)`` uint8 image.

    Returns:
        np.ndarray: ``(480, 832, C)`` uint8 image.

    Raises:
        ValueError: if the frame is not 720×1280.
    """
    import cv2

    if frame.shape[:2] != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, crop assumes 1280x720")
    margin = (SOURCE_WIDTH - CROP_WIDTH) // 2
    cropped = frame[:, margin : margin + CROP_WIDTH]
    return cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)


def decode_clip_frames(video_path: str, source_frames: np.ndarray) -> np.ndarray:
    """Decode the planned source frames of a clip, cropped and resized to 480×832.

    Frames are decoded sequentially from the start of the video and counted in presentation
    order, so the frame numbers match ``indexes.txt`` without relying on seeking. This decodes
    ``source_frames.max() + 1`` frames, also when the window starts later in the clip.

    Args:
        video_path (str): the clip's mp4.
        source_frames (np.ndarray): ``(T,)`` source frame numbers.

    Returns:
        np.ndarray: ``(T, 480, 832, 3)`` uint8 RGB.

    Raises:
        ValueError: if the video ends before the last requested frame.
    """
    import av

    wanted: dict[int, list[int]] = {}
    for pos, index in enumerate(np.asarray(source_frames).tolist()):
        wanted.setdefault(index, []).append(pos)
    last = max(wanted)
    out = np.empty((len(source_frames), OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    count = 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            count = index + 1
            if index in wanted:
                image = crop_resize_frame(frame.to_ndarray(format="rgb24"))
                for pos in wanted[index]:
                    out[pos] = image
            if index == last:
                return out
    raise ValueError(f"{video_path}: {count} frames decoded, need frame {last}")


def load_caption(ann_dir: str, field: str) -> str:
    """Read one caption field from a clip's ``caption.json``.

    Args:
        ann_dir (str): the clip's annotation directory.
        field (str): one of ``CAPTION_FIELDS``.

    Returns:
        str: the caption text.

    Raises:
        ValueError: if ``field`` is not a caption field or the clip has no text for it.
    """
    import json

    if field not in CAPTION_FIELDS:
        raise ValueError(f"caption field {field!r} is not one of {CAPTION_FIELDS}")
    with open(os.path.join(ann_dir, "caption.json")) as f:
        text = json.load(f).get(field)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{ann_dir}: caption.json has no {field}")
    return text.strip()


def _summarize(plan: ClipPlan, meta: Mapping) -> str:
    path = "n/a" if plan.window_path is None else f"{plan.window_path:.3f}"
    head = (
        f"{plan.clip_id} fps={float(meta['fps']):.3f} start={plan.start_frame} "
        f"window_path_m={path} dynamicRatio={float(meta['dynamicRatio']):.3f}"
    )
    if plan.skipped:
        return f"{head} SKIP {','.join(plan.skip_reasons)}"
    centres = invert_rigid(pose_vec_to_matrix(plan.poses))[:, :3, 3] / TRANSLATION_SCALE
    angle = np.degrees(2 * np.arccos(np.clip(abs(plan.poses[-1, 6]), 0, 1)))
    return (
        f"{head} intrinsics={np.round(plan.intrinsics, 4).tolist()} "
        f"end_centre_m={np.round(centres[-1], 3).tolist()} "
        f"end_rot_deg={angle:.1f} depth_indices={plan.depth_indices.tolist()}"
    )


def main(argv: list[str] | None = None) -> None:
    """Plan the clips of one group, print a summary per clip, optionally save the kept ids.

    A clip whose annotation breaks the contract (e.g. ``indexes.txt`` and ``poses.npy`` of
    different lengths) is printed as an error, counted apart from the skips and left out of
    ``--output``.

    Args:
        argv (list[str] | None): command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", required=True, help="SpatialVID_HQ_metadata.csv")
    parser.add_argument(
        "--annotations_root",
        required=True,
        help="directory that the metadata 'annotation path' column is relative to",
    )
    parser.add_argument(
        "--videos_root",
        required=True,
        help="directory that the metadata 'video path' column is relative to",
    )
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--num_clips", type=int, default=None, help="first N ids; default all")
    parser.add_argument("--min_window_path", type=float, default=MIN_WINDOW_PATH)
    parser.add_argument("--max_dynamic_ratio", type=float, default=MAX_DYNAMIC_RATIO)
    parser.add_argument("--output", default=None, help="write kept clip ids here, one per line")
    args = parser.parse_args(argv)

    rows = read_metadata(args.metadata, group=args.group)
    clip_ids = sorted(rows)[: args.num_clips]
    kept, skipped, errors = [], 0, 0
    for clip_id in clip_ids:
        row = rows[clip_id]
        try:
            annotation = load_annotation(
                os.path.join(args.annotations_root, row["annotation path"])
            )
            plan = plan_clip(
                annotation,
                row,
                hdr=video_is_hdr(os.path.join(args.videos_root, row["video path"])),
                min_window_path=args.min_window_path,
                max_dynamic_ratio=args.max_dynamic_ratio,
            )
        except ValueError as e:
            print(f"{clip_id} ERROR {e}")
            errors += 1
            continue
        print(_summarize(plan, row))
        if plan.skipped:
            skipped += 1
        else:
            kept.append(clip_id)
    print(f"kept {len(kept)}, skipped {skipped}, errors {errors} of {len(clip_ids)}")
    if args.output:
        with open(args.output, "w") as f:
            f.writelines(f"{clip_id}\n" for clip_id in kept)


if __name__ == "__main__":
    main()
