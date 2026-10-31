"""Tests for the WorldScore bridge in ``tools/eval/worldscore``.

WorldScore itself is not imported: its single-move cameras are rebuilt here the
way its ``CameraGen`` builds them (PyTorch3D-style w2c ``[R|T]``, slerp/lerp per
pixel frame, saved as Blender c2w), so the pose conversion is checked against
minWM's own trajectory strings.
"""

import json
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from minwm.processors.camera import parse_trajectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "eval", "worldscore"))

import _common  # noqa: E402
import pack  # noqa: E402
import prepare  # noqa: E402
import summarize  # noqa: E402


def _yaw(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])


def _rt(rot: np.ndarray, trans) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, trans
    return m


def _worldscore_cameras(layout: str) -> np.ndarray:
    """WorldScore's ``cameras_interp`` for a one-move camera path."""
    orbit = {"orbit_left": -30.0, "orbit_right": 30.0}
    end = {
        "push_in": _rt(np.eye(3), [0, 0, -1]),
        "pull_out": _rt(np.eye(3), [0, 0, 1]),
        "move_left": _rt(np.eye(3), [-0.5, 0, 0]),
        "move_right": _rt(np.eye(3), [0.5, 0, 0]),
        "pan_left": _rt(_yaw(30), [0, 0, 0]),
        "pan_right": _rt(_yaw(-30), [0, 0, 0]),
    }
    if layout in orbit:
        end[layout] = _rt(np.eye(3), [0, 0, 1]) @ _rt(_yaw(orbit[layout]), [0, 0, 0])
        end[layout] = end[layout] @ _rt(np.eye(3), [0, 0, -1])
    r_end, t_end = end[layout][:3, :3], end[layout][:3, 3]
    slerp = Slerp([0, 1], Rotation.from_matrix([np.eye(3), r_end]))
    n = _common.NUM_PIXEL_FRAMES - 1
    w2c_pt3d = [_rt(slerp([i / n]).as_matrix()[0], i / n * t_end) for i in range(n + 1)]
    pt3d_to_blender = np.diag([-1.0, 1.0, -1.0, 1.0])
    return np.stack([np.linalg.inv(pt3d_to_blender @ m) for m in w2c_pt3d])


@pytest.mark.parametrize("layout", sorted(_common._EXPECTED_MOTION.keys() - {"fixed"}))
def test_converted_poses_start_at_identity_and_move_as_named(layout):
    poses = _common.worldscore_to_minwm_poses(_worldscore_cameras(layout), translation_scale=1.52)
    assert poses.shape == (_common.NUM_LATENT_FRAMES, 4, 4)
    assert np.allclose(poses[0], np.eye(4), atol=1e-6)
    _common.check_motion(layout, poses, translation_scale=1.52)


def test_pan_left_matches_minwm_yaw_left_string():
    poses = _common.worldscore_to_minwm_poses(_worldscore_cameras("pan_left"))
    # 10 steps of 3 deg = WorldScore's 30 deg pan.
    assert np.allclose(poses[-1], parse_trajectory("j*10")[-1], atol=1e-5)


def test_push_in_at_default_scale_matches_minwm_forward_string():
    poses = _common.worldscore_to_minwm_poses(
        _worldscore_cameras("push_in"), translation_scale=1.52
    )
    assert np.allclose(poses, parse_trajectory("w*19"), atol=1e-5)


def test_check_motion_catches_a_mirrored_camera():
    poses = _common.worldscore_to_minwm_poses(_worldscore_cameras("pan_left"))
    with pytest.raises(ValueError, match="forward x"):
        _common.check_motion("pan_right", poses)


def test_conversion_rejects_wrong_frame_count():
    with pytest.raises(ValueError, match="cameras"):
        _common.worldscore_to_minwm_poses(np.tile(np.eye(4), (81, 1, 1)))


def test_instance_id_round_trips_and_rejects_unsafe_paths():
    rel = os.path.join("static", "photorealistic", "indoor", "dining_spaces", "000")
    assert _common.instance_id(rel) == "static--photorealistic--indoor--dining_spaces--000"
    assert _common.instance_rel_path(_common.instance_id(rel)) == rel
    with pytest.raises(ValueError):
        _common.instance_id(os.path.join("static", "a b", "000"))


def test_instance_seed_is_fixed_by_id():
    a = "static--photorealistic--indoor--dining_spaces--000"
    assert _common.instance_seed(a) == _common.instance_seed(a)
    assert _common.instance_seed(a) != _common.instance_seed(a[:-1] + "1")
    assert 0 <= _common.instance_seed(a) < 2**31


def _items(groups: str = "abc", per: int = 10) -> list[dict]:
    return [{"image": f"{g}/{i:03d}.jpg", "g": g} for g in groups for i in range(per)]


def _group(x: dict) -> tuple:
    return (x["g"],)


def test_select_subset_is_stratified_rank_major_and_seeded():
    first = prepare.select_subset(_items(), _group, per_group=3, seed=0)
    assert [x["g"] for x in first] == list("abcabcabc")
    assert first == prepare.select_subset(list(reversed(_items())), _group, 3, 0)
    assert first != prepare.select_subset(_items(), _group, 3, 1)
    assert len(prepare.select_subset(_items(), _group, per_group=50, seed=0)) == 30


def test_select_subset_is_nested_across_sizes_and_groups():
    small = prepare.select_subset(_items("ab"), _group, per_group=2, seed=0)
    large = prepare.select_subset(_items("abc"), _group, per_group=6, seed=0)
    assert all(item in large for item in small)


def test_write_subset_may_grow_but_not_shrink(tmp_path):
    path = str(tmp_path / "subset" / "static.json")
    small = prepare.select_subset(_items(), _group, per_group=2, seed=0)
    large = prepare.select_subset(_items(), _group, per_group=4, seed=0)
    prepare.write_subset(path, small)
    prepare.write_subset(path, large)
    with open(path) as f:
        assert json.load(f) == large
    with pytest.raises(SystemExit, match="only grow"):
        prepare.write_subset(path, small)


def test_unknown_move_is_rejected(monkeypatch):
    argv = ["prepare.py", "--worldscore-root", "w", "--dataset-root", "d", "--runs-root", "r"]
    monkeypatch.setattr(sys, "argv", argv + ["--bench", "b", "--moves", "push_in,zoom"])
    with pytest.raises(SystemExit):
        prepare.parse_args()


def test_summary_without_dynamic_split_has_no_dynamic_score(tmp_path, monkeypatch):
    instance = tmp_path / _common.OUTPUT_DIR / "static" / "photorealistic" / "indoor" / "c" / "000"
    instance.mkdir(parents=True)
    (instance / "image_data.json").write_text(json.dumps({"camera_path": ["pan_left"]}))
    evaluation = {"3d_consistency": {"reprojection_error": {"score": 0.2, "score_normalized": 0.9}}}
    (instance / "evaluation.json").write_text(json.dumps(evaluation))
    monkeypatch.setattr(sys, "argv", ["summarize.py", "--runs-root", str(tmp_path)])
    summarize.main()
    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)
    assert list(summary["overall"]) == ["WorldScore-Static"]
    assert "动态" not in (tmp_path / "summary.md").read_text()
    pan = next(row for row in summary["static"] if row["group"] == "pan_left")
    assert pan["3d_consistency"] == pytest.approx(90.0)


def test_summary_follows_worldscore_aggregation_and_skips_sentinels():
    evals = [
        {
            "subjective_quality": {
                "clip_iqa+": {"score": 0.6, "score_normalized": 0.4},
                "clip_aesthetic": {"score": 5.0, "score_normalized": 0.8},
            },
            "3d_consistency": {"reprojection_error": {"score": 0.2, "score_normalized": 0.9}},
        },
        {
            "subjective_quality": {
                "clip_iqa+": {"score": 0.6, "score_normalized": 0.6},
                "clip_aesthetic": {"score": 5.0, "score_normalized": 0.8},
            },
            "3d_consistency": {"reprojection_error": {"score": 100, "score_normalized": 0.0}},
        },
    ]
    scores = summarize.aspect_scores(
        evals, ("subjective_quality", "3d_consistency", "style_consistency")
    )
    assert scores["subjective_quality"] == pytest.approx(100 * (0.5 + 0.8) / 2)
    assert scores["3d_consistency"] == pytest.approx(45.0)
    assert scores["style_consistency"] is None
    means, sentinels = summarize.raw_means(
        evals, {"reproj": ("3d_consistency", "reprojection_error", None)}
    )
    assert means["reproj"] == pytest.approx(0.2)
    assert sentinels == 1


def _fake_run(tmp_path, num_frames: int) -> tuple[str, str, str]:
    runs_root, gen_dir = str(tmp_path / "run"), str(tmp_path / "run" / "gen")
    rel = os.path.join("dynamic", "photorealistic", "fluid", "007")
    instance = os.path.join(runs_root, _common.OUTPUT_DIR, rel)
    os.makedirs(instance)
    with open(os.path.join(instance, "image_data.json"), "w") as f:
        json.dump({"total_frames": 3}, f)
    sample_id = _common.instance_id(rel)
    os.makedirs(os.path.join(gen_dir, f"{sample_id}.frames"))
    for i in range(num_frames):
        open(os.path.join(gen_dir, f"{sample_id}.frames", f"{i:05d}.png"), "wb").close()
    open(os.path.join(gen_dir, f"{sample_id}.mp4"), "wb").close()
    return runs_root, gen_dir, instance


def test_pack_moves_frames_and_video_into_the_instance(tmp_path, monkeypatch):
    runs_root, gen_dir, instance = _fake_run(tmp_path, num_frames=3)
    monkeypatch.setattr(sys, "argv", ["pack.py", "--gen-dir", gen_dir, "--runs-root", runs_root])
    pack.main()
    assert sorted(os.listdir(os.path.join(instance, "frames"))) == [
        "00000.png",
        "00001.png",
        "00002.png",
    ]
    assert os.path.exists(os.path.join(instance, "videos", "output.mp4"))
    assert os.listdir(gen_dir) == []


def test_pack_refuses_a_frame_count_worldscore_would_reject(tmp_path, monkeypatch):
    runs_root, gen_dir, _ = _fake_run(tmp_path, num_frames=2)
    monkeypatch.setattr(sys, "argv", ["pack.py", "--gen-dir", gen_dir, "--runs-root", runs_root])
    with pytest.raises(SystemExit, match="expects 3"):
        pack.main()
