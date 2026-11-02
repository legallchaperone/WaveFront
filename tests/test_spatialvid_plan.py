"""CPU tests for the SpatialVID clip planner on synthetic annotations."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.data.wan21.spatialvid import (
    MIN_SPAN_SECONDS,
    NUM_LATENT_FRAMES,
    NUM_PIXEL_FRAMES,
    SKIP_LONG_FOCAL,
    SKIP_NEAR_STATIC,
    SKIP_SHORT_SPAN,
    SKIP_TOO_DYNAMIC,
    TRANSLATION_SCALE,
    ClipAnnotation,
    crop_intrinsics,
    interpolate_c2w,
    invert_rigid,
    latent_frame_poses,
    load_annotation,
    matrix_to_pose_vec,
    nearest_annotation,
    output_source_frames,
    plan_clip,
    pose_vec_to_matrix,
)

KEEP_ALL = dict(min_move_dist=0.0, max_dynamic_ratio=1.0)
INTRINSICS = [0.5, 0.5 * 16 / 9, 0.5, 0.5]


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rigid(rot: np.ndarray, t) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = t
    return m


def _annotation(c2w: np.ndarray, frames, intrinsics=None, clip_id="clip") -> ClipAnnotation:
    """SpatialVID-style annotation (w2c pose rows) of a c2w trajectory."""
    n = len(frames)
    intr = np.tile(intrinsics or INTRINSICS, (n, 1)).astype(np.float64)
    return ClipAnnotation(
        clip_id=clip_id,
        indices=np.arange(n, dtype=np.int64),
        frames=np.asarray(frames, dtype=np.int64),
        poses=matrix_to_pose_vec(invert_rigid(c2w)),
        intrinsics=intr,
    )


def _meta(clip_id="clip", fps=30.0, move_dist=1.0, dynamic_ratio=0.1, resolution="1280x720"):
    return {
        "id": clip_id,
        "fps": str(fps),
        "resolution": resolution,
        "moveDist": str(move_dist),
        "dynamicRatio": str(dynamic_ratio),
    }


def _forward_trajectory(frames, fps, speed=0.5, yaw_deg_per_s=0.0, start=None):
    """c2w poses at ``frames``: moving forward along world +z while yawing about world y."""
    start = np.eye(4) if start is None else start
    out = []
    for f in frames:
        t = f / fps
        out.append(start @ _rigid(_rot_y(yaw_deg_per_s * t), [0.0, 0.0, speed * t]))
    return np.stack(out)


# --- output frame times ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fps, head, last",
    [
        (24.0, [0, 2, 3, 5, 6, 8, 9, 11, 12], 66),
        (30.0, [0, 2, 4, 6, 8, 9, 11, 13, 15], 83),
        (60.0, [0, 4, 8, 11, 15, 19, 23, 26, 30], 165),
    ],
)
def test_output_source_frames(fps, head, last):
    frames = output_source_frames(fps)
    assert frames.shape == (NUM_PIXEL_FRAMES,)
    assert frames[: len(head)].tolist() == head
    assert frames[-1] == last
    j = np.arange(NUM_PIXEL_FRAMES)
    np.testing.assert_array_equal(frames, np.floor(j * fps / 16 + 0.5))


def test_source_times_are_frame_over_fps():
    fps = 29.97
    ann = _annotation(_forward_trajectory(range(0, 100, 5), fps), range(0, 100, 5))
    plan = plan_clip(ann, _meta(fps=fps), **KEEP_ALL)
    np.testing.assert_allclose(plan.source_times, output_source_frames(fps) / fps)
    assert abs(plan.source_times[4] - 0.25) < 0.5 / fps


# --- interpolation -----------------------------------------------------------------------------


def test_interpolation_hits_annotation_poses_and_midpoint():
    c2w = np.stack([_rigid(_rot_y(0), [0, 0, 0]), _rigid(_rot_y(40), [2, 0, 4])])
    frames = np.array([0, 10])
    out = interpolate_c2w(c2w, frames, np.array([0, 5, 10]))
    np.testing.assert_allclose(out[0], c2w[0], atol=1e-12)
    np.testing.assert_allclose(out[2], c2w[1], atol=1e-12)
    np.testing.assert_allclose(out[1], _rigid(_rot_y(20), [1, 0, 2]), atol=1e-12)


def test_interpolation_uses_neighbouring_annotation_frames_only():
    c2w = np.stack(
        [_rigid(_rot_y(0), [0, 0, 0]), _rigid(_rot_y(30), [0, 0, 1]), _rigid(_rot_y(90), [5, 0, 1])]
    )
    out = interpolate_c2w(c2w, np.array([0, 4, 12]), np.array([2.0, 8.0]))
    np.testing.assert_allclose(out[0], _rigid(_rot_y(15), [0, 0, 0.5]), atol=1e-12)
    np.testing.assert_allclose(out[1], _rigid(_rot_y(60), [2.5, 0, 1]), atol=1e-12)


def test_interpolation_refuses_to_extrapolate():
    c2w = np.stack([np.eye(4), np.eye(4)])
    with pytest.raises(ValueError, match="outside"):
        interpolate_c2w(c2w, np.array([0, 10]), np.array([11]))


def test_slerp_keeps_unit_quaternions():
    rng = np.random.default_rng(0)
    rots = Rotation.random(6, random_state=1).as_matrix()
    c2w = np.stack([_rigid(r, rng.normal(size=3)) for r in rots])
    out = interpolate_c2w(c2w, np.arange(6) * 7, np.linspace(0, 35, 200))
    quats = matrix_to_pose_vec(out)[:, 3:]
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)
    rot = out[:, :3, :3]
    np.testing.assert_allclose(
        rot @ np.swapaxes(rot, 1, 2), np.tile(np.eye(3), (200, 1, 1)), atol=1e-12
    )


# --- latent-frame poses ------------------------------------------------------------------------


def test_alignment_makes_frame0_identity():
    start = _rigid(_rot_x(25) @ _rot_y(-70), [3.0, -1.0, 7.0])
    frames = np.arange(0, 100, 5)
    ann = _annotation(_forward_trajectory(frames, 30.0, yaw_deg_per_s=10, start=start), frames)
    poses = latent_frame_poses(ann, 30.0)
    assert poses.shape == (NUM_LATENT_FRAMES, 7) and poses.dtype == np.float32
    np.testing.assert_allclose(poses[0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(abs(poses[0, 6]), 1.0, atol=1e-6)
    # The relative motion survives alignment: same as the trajectory started at the identity.
    ref = latent_frame_poses(
        _annotation(_forward_trajectory(frames, 30.0, yaw_deg_per_s=10), frames), 30.0
    )
    np.testing.assert_allclose(pose_vec_to_matrix(poses), pose_vec_to_matrix(ref), atol=1e-5)


def test_translation_scale_applied():
    fps, speed = 24.0, 0.4
    frames = np.arange(0, 80, 4)
    ann = _annotation(_forward_trajectory(frames, fps, speed=speed), frames)
    poses = latent_frame_poses(ann, fps)
    times = output_source_frames(fps)[::4] / fps
    # Pure forward motion with identity rotation: w2c translation is −scale·centre.
    np.testing.assert_allclose(poses[:, 2], -TRANSLATION_SCALE * speed * times, atol=1e-6)
    np.testing.assert_allclose(poses[:, :2], 0.0, atol=1e-7)


def test_known_c2w_converts_to_expected_w2c():
    # Camera yaws right by 90° per second about +y and moves to +x; at fps 16 and annotations
    # every 4 frames, latent frame k sits exactly on annotation frame k.
    fps = 16.0
    frames = np.arange(0, 48, 4)
    c2w = np.stack([_rigid(_rot_y(90 * f / fps), [f / fps, 0, 0]) for f in frames])
    poses = latent_frame_poses(_annotation(c2w, frames), fps)
    for k, f in enumerate(frames):
        theta = np.radians(90 * f / fps)
        centre = np.array([f / fps, 0, 0]) * TRANSLATION_SCALE
        # w2c = [R_y(θ)ᵀ | −R_y(θ)ᵀ c]; R_y(θ)ᵀ = R_y(−θ) has quaternion (0, −sin θ/2, 0, cos θ/2).
        t_expected = -_rot_y(np.degrees(theta)).T @ centre
        q_expected = np.array([0, -np.sin(theta / 2), 0, np.cos(theta / 2)])
        q = poses[k, 3:]
        np.testing.assert_allclose(poses[k, :3], t_expected, atol=1e-5)
        assert np.allclose(q, q_expected, atol=1e-6) or np.allclose(q, -q_expected, atol=1e-6)


def test_latent_poses_interpolate_between_annotation_frames():
    # 30 fps, annotations every 5 frames: latent frame 1 is source frame 8, between 5 and 10.
    fps, speed = 30.0, 0.6
    frames = np.arange(0, 100, 5)
    ann = _annotation(_forward_trajectory(frames, fps, speed=speed, yaw_deg_per_s=12), frames)
    poses = pose_vec_to_matrix(latent_frame_poses(ann, fps))
    c2w = invert_rigid(poses)
    assert output_source_frames(fps)[4] == 8
    np.testing.assert_allclose(c2w[1, 2, 3], TRANSLATION_SCALE * speed * 8 / fps, atol=1e-6)
    np.testing.assert_allclose(c2w[1, :3, :3], _rot_y(12 * 8 / fps), atol=1e-6)


# --- intrinsics --------------------------------------------------------------------------------


def test_crop_intrinsics_formula():
    out = crop_intrinsics(np.array(INTRINSICS))
    assert out.dtype == np.float32 and out.shape == (4,)
    np.testing.assert_allclose(out, [0.5 * 1280 / 1248, 0.5 * 16 / 9, 0.5, 0.5], rtol=1e-6)
    off = crop_intrinsics(np.array([0.7, 1.2, 0.6, 0.45]))
    np.testing.assert_allclose(
        off, [0.7 * 1280 / 1248, 1.2, (0.6 * 1280 - 16) / 1248, 0.45], rtol=1e-6
    )


# --- skip reasons and errors -------------------------------------------------------------------


def _plan(fps=30.0, last_frame=90, step=5, intrinsics=None, **meta):
    frames = np.arange(0, last_frame + 1, step)
    ann = _annotation(_forward_trajectory(frames, fps), frames, intrinsics=intrinsics)
    return plan_clip(ann, _meta(fps=fps, **meta), min_move_dist=0.05, max_dynamic_ratio=0.3)


def test_kept_clip_has_full_plan():
    plan = _plan()
    assert not plan.skipped and plan.skip_reasons == ()
    assert plan.source_frames.shape == (NUM_PIXEL_FRAMES,)
    assert plan.poses.shape == (NUM_LATENT_FRAMES, 7) and plan.poses.dtype == np.float32
    assert plan.intrinsics.shape == (4,) and plan.intrinsics.dtype == np.float32
    assert plan.depth_indices.shape == (NUM_LATENT_FRAMES,)


def test_skip_short_span_and_boundary():
    # 24 fps: a span of exactly 2.75 s ends on source frame 66, which pixel frame 44 needs.
    assert MIN_SPAN_SECONDS * 24 == 66
    assert not _plan(fps=24.0, last_frame=66, step=6).skipped
    assert _plan(fps=24.0, last_frame=60, step=6).skip_reasons == (SKIP_SHORT_SPAN,)


def test_skip_long_focal():
    # Cropped fx/W = fx·1280/1248: 1.46 -> 1.497 kept, 1.47 -> 1.508 skipped.
    assert not _plan(intrinsics=[1.46, 2.6, 0.5, 0.5]).skipped
    assert _plan(intrinsics=[1.47, 2.6, 0.5, 0.5]).skip_reasons == (SKIP_LONG_FOCAL,)


def test_skip_metadata_filters_and_multiple_reasons():
    assert _plan(move_dist=0.01).skip_reasons == (SKIP_NEAR_STATIC,)
    assert _plan(dynamic_ratio=0.5).skip_reasons == (SKIP_TOO_DYNAMIC,)
    plan = _plan(last_frame=60, move_dist=0.0, dynamic_ratio=0.9)
    assert plan.skip_reasons == (SKIP_SHORT_SPAN, SKIP_NEAR_STATIC, SKIP_TOO_DYNAMIC)
    assert plan.poses is None and plan.intrinsics is None and plan.depth_indices is None


def test_metadata_thresholds_have_no_default():
    frames = np.arange(0, 91, 5)
    ann = _annotation(_forward_trajectory(frames, 30.0), frames)
    with pytest.raises(TypeError):
        plan_clip(ann, _meta())


def test_varying_intrinsics_is_an_error():
    frames = np.arange(0, 91, 5)
    ann = _annotation(_forward_trajectory(frames, 30.0), frames)
    ann.intrinsics[3, 0] += 1e-4
    with pytest.raises(ValueError, match="intrinsics vary"):
        plan_clip(ann, _meta(), **KEEP_ALL)


@pytest.mark.parametrize(
    "frames, meta, match",
    [
        (np.arange(5, 96, 5), {}, "not 0"),
        (np.arange(0, 91, 5), {"resolution": "1920x1080"}, "resolution"),
        (np.arange(0, 91, 5), {"clip_id": "other"}, "does not belong"),
    ],
)
def test_contract_violations_are_errors(frames, meta, match):
    ann = _annotation(_forward_trajectory(frames, 30.0), frames)
    with pytest.raises(ValueError, match=match):
        plan_clip(ann, _meta(**meta), **KEEP_ALL)


# --- depth frame choice ------------------------------------------------------------------------


def test_nearest_annotation_and_ties_go_earlier():
    frames = np.array([0, 5, 10, 15, 20])
    np.testing.assert_array_equal(
        nearest_annotation(frames, np.array([0, 8, 12, 20])), [0, 2, 2, 4]
    )
    np.testing.assert_array_equal(nearest_annotation(np.array([0, 4, 8]), np.array([2, 6])), [0, 1])


def test_depth_indices_are_annotation_column0():
    fps = 30.0
    frames = np.arange(0, 100, 5)
    ann = _annotation(_forward_trajectory(frames, fps), frames)
    ann = ClipAnnotation(ann.clip_id, ann.indices + 100, ann.frames, ann.poses, ann.intrinsics)
    plan = plan_clip(ann, _meta(fps=fps), **KEEP_ALL)
    # Latent source frames 0, 8, 15, 23, 30, ... -> nearest annotation frames 0, 10, 15, 25, 30.
    assert plan.depth_indices[:5].tolist() == [100, 102, 103, 105, 106]


# --- file loading ------------------------------------------------------------------------------


def test_load_annotation_reads_spatialvid_layout(tmp_path):
    fps = 30.0
    frames = np.arange(0, 91, 6)
    ann = _annotation(_forward_trajectory(frames, fps, yaw_deg_per_s=20), frames)
    clip_dir = tmp_path / "abc"
    clip_dir.mkdir()
    lines = [f"# total {len(frames)} indexes"] + [f"{i} {f}" for i, f in enumerate(frames)]
    (clip_dir / "indexes.txt").write_text("\n".join(lines) + "\n")
    np.save(clip_dir / "poses.npy", ann.poses.astype(np.float32))
    np.save(clip_dir / "intrinsics.npy", ann.intrinsics.astype(np.float32))

    loaded = load_annotation(str(clip_dir))
    assert loaded.clip_id == "abc"
    np.testing.assert_array_equal(loaded.frames, frames)
    np.testing.assert_array_equal(loaded.indices, np.arange(len(frames)))
    np.testing.assert_allclose(loaded.poses, ann.poses, atol=1e-6)
    plan = plan_clip(loaded, _meta(clip_id="abc", fps=fps), **KEEP_ALL)
    assert not plan.skipped


def test_load_annotation_length_mismatch(tmp_path):
    (tmp_path / "indexes.txt").write_text("0 0\n1 5\n")
    np.save(tmp_path / "poses.npy", np.zeros((3, 7), np.float32))
    np.save(tmp_path / "intrinsics.npy", np.zeros((2, 4), np.float32))
    with pytest.raises(ValueError, match="2 indexes, 3 poses"):
        load_annotation(str(tmp_path))
