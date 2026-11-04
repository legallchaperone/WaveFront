"""Tests for the WorldScore bridge in ``tools/eval/worldscore``.

WorldScore itself is not imported: its single-move cameras are rebuilt here the
way its ``CameraGen`` builds them (PyTorch3D-style w2c ``[R|T]``, slerp/lerp per
pixel frame, saved as Blender c2w), so the pose conversion is checked against
minWM's own trajectory strings.
"""

import json
import os
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from minwm.processors.camera import parse_trajectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "eval", "worldscore"))

import _common  # noqa: E402
import pack  # noqa: E402
import prepare  # noqa: E402
import report  # noqa: E402
import score_disparity  # noqa: E402
import score_vbench  # noqa: E402
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


def test_worldscore_intrinsics_crop_the_square_camera_to_minwm_frames():
    fx, fy, cx, cy = _common.worldscore_intrinsics(500.0)
    # A square source is too tall for 832x480: the crop keeps the full width.
    assert fx == pytest.approx(500 / 512)
    assert fy == pytest.approx(832 / 480 * 500 / 512)
    assert (cx, cy) == (0.5, 0.5)


def test_worldscore_intrinsics_crop_the_width_of_a_wider_source():
    fx, fy, _, _ = _common.worldscore_intrinsics(500.0, source_size=(1280, 480))
    assert fy == pytest.approx(500 / 480)
    assert fx == pytest.approx(500 / 832)


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


def test_clip_downloads_default_to_the_torch_hub_dir(monkeypatch, tmp_path):
    import types

    import torch

    fake = types.ModuleType("clip")
    fake.load = lambda name, device="cpu", jit=False, download_root=None: download_root
    monkeypatch.setitem(sys.modules, "clip", fake)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    _common.keep_clip_downloads_off_home()
    assert fake.load("ViT-L/14") == os.path.join(torch.hub.get_dir(), "clip")
    assert torch.hub.get_dir().startswith(str(tmp_path))
    assert fake.load("ViT-L/14", download_root="/elsewhere") == "/elsewhere"


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
    open(os.path.join(gen_dir, f"{sample_id}.disparity.pt"), "wb").close()
    return runs_root, gen_dir, instance


def test_pack_moves_frames_video_and_disparity_into_the_instance(tmp_path, monkeypatch):
    runs_root, gen_dir, instance = _fake_run(tmp_path, num_frames=3)
    monkeypatch.setattr(sys, "argv", ["pack.py", "--gen-dir", gen_dir, "--runs-root", runs_root])
    pack.main()
    assert sorted(os.listdir(os.path.join(instance, "frames"))) == [
        "00000.png",
        "00001.png",
        "00002.png",
    ]
    assert os.path.exists(os.path.join(instance, "videos", "output.mp4"))
    assert os.path.exists(os.path.join(instance, "disparity.pt"))
    assert os.listdir(gen_dir) == []


def test_pack_refuses_a_frame_count_worldscore_would_reject(tmp_path, monkeypatch):
    runs_root, gen_dir, _ = _fake_run(tmp_path, num_frames=2)
    monkeypatch.setattr(sys, "argv", ["pack.py", "--gen-dir", gen_dir, "--runs-root", runs_root])
    with pytest.raises(SystemExit, match="expects 3"):
        pack.main()


def _packed_instance(runs_root: str, rel: str) -> str:
    instance = os.path.join(runs_root, _common.OUTPUT_DIR, rel)
    os.makedirs(os.path.join(instance, "videos"))
    open(os.path.join(instance, "image_data.json"), "w").close()
    open(os.path.join(instance, "videos", "output.mp4"), "wb").close()
    return instance


class _FakeVBench:
    """Stands in for ``vbench.VBench``: writes ``*_eval_results.json`` as VBench does."""

    calls = []

    def __init__(self, device, full_info_dir, output_path):
        self.output_path = output_path

    def evaluate(self, videos_path, name, dimension_list, local, mode):
        videos = sorted(os.listdir(videos_path))
        type(self).calls.append((videos, dimension_list, mode))
        per_video = {
            "imaging_quality": 62.0,  # VBench keeps per-video MUSIQ on its 0-100 scale
            "subject_consistency": 0.9,
        }
        results = {
            dim: [
                0.0,
                [
                    {"video_path": os.path.join(videos_path, v), "video_results": per_video[dim]}
                    for v in videos
                ],
            ]
            for dim in dimension_list
        }
        with open(os.path.join(self.output_path, f"{name}_eval_results.json"), "w") as f:
            json.dump(results, f)


def test_score_vbench_writes_per_instance_scores_and_resumes(tmp_path, monkeypatch):
    runs_root = str(tmp_path / "run")
    rels = [
        os.path.join("static", "photorealistic", "indoor", "dining_spaces", "000"),
        os.path.join("static", "photorealistic", "outdoor", "city", "001"),
    ]
    instances = [_packed_instance(runs_root, rel) for rel in rels]
    # An instance without a video is not generated yet and must be left alone.
    ungenerated = os.path.join(runs_root, _common.OUTPUT_DIR, "static", "p", "indoor", "c", "002")
    os.makedirs(ungenerated)
    open(os.path.join(ungenerated, "image_data.json"), "w").close()
    fake = types.ModuleType("vbench")
    fake.__file__ = str(tmp_path / "vbench" / "__init__.py")
    fake.VBench = _FakeVBench
    monkeypatch.setitem(sys.modules, "vbench", fake)
    _FakeVBench.calls.clear()
    argv = ["score_vbench.py", "--runs-root", runs_root]
    dims = "--dimensions=imaging_quality,subject_consistency"

    monkeypatch.setattr(sys, "argv", [*argv, "--dimensions=imaging_quality"])
    score_vbench.main()
    ((videos, _, mode),) = _FakeVBench.calls
    assert mode == "custom_input"
    assert videos == sorted(f"{_common.instance_id(rel)}.mp4" for rel in rels)
    for instance in instances:
        with open(os.path.join(instance, "vbench.json")) as f:
            assert json.load(f) == {"imaging_quality": pytest.approx(0.62)}

    monkeypatch.setattr(sys, "argv", [*argv, dims])
    score_vbench.main()
    score_vbench.main()  # everything scored: VBench is not called again
    assert len(_FakeVBench.calls) == 2
    with open(os.path.join(instances[0], "vbench.json")) as f:
        assert json.load(f) == {"imaging_quality": 0.62, "subject_consistency": 0.9}
    assert not [p for p in os.listdir(runs_root) if p.startswith("vbench-")]


def _disparity(seed: int = 0, shape=(48, 64)) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, shape)


def test_affine_fit_recovers_an_affine_invariant_estimate():
    d = _disparity()
    estimate = (d - 0.3) / 2.0  # an estimator's inverse depth with its own scale and shift
    m = score_disparity.frame_metrics(d, estimate, "affine")
    assert m["scale"] == pytest.approx(2.0) and m["shift"] == pytest.approx(0.3)
    assert m["abs_rel"] == pytest.approx(0.0, abs=1e-9) and m["delta1"] == 1.0
    # A scale-only fit charges the estimator's shift to the model.
    assert score_disparity.frame_metrics(d, estimate, "scale")["abs_rel"] > 0.1


def test_scale_fit_is_exact_for_an_up_to_scale_estimate():
    d = _disparity()
    m = score_disparity.frame_metrics(d, d / 3.0, "scale")
    assert m["scale"] == pytest.approx(3.0) and m["shift"] == 0.0
    assert m["abs_rel"] == pytest.approx(0.0, abs=1e-9)


def test_saturated_and_far_pixels_are_left_out():
    d = _disparity()
    d[:16] = 1.0  # clipped: the true disparity is unknown
    d[16:24] = 0.01  # beyond the floor
    m = score_disparity.frame_metrics(d, d.copy(), "affine")
    assert m["valid_fraction"] == pytest.approx(24 / 48)
    assert m["abs_rel"] == pytest.approx(0.0, abs=1e-9)


def test_too_few_pixels_give_no_score():
    d = np.full((8, 8), 1.0)
    m = score_disparity.frame_metrics(d, d, "affine")
    assert m["abs_rel"] is None and m["delta1"] is None and m["valid_fraction"] == 0.0


def test_score_disparity_compares_latent_frame_k_with_pixel_frame_4k(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    runs_root = str(tmp_path / "run")
    with_d = _packed_instance(runs_root, os.path.join("static", "p", "indoor", "c", "000"))
    without_d = _packed_instance(runs_root, os.path.join("static", "p", "indoor", "c", "001"))
    d = np.stack([_disparity(seed=k) for k in range(3)]).astype(np.float32)
    torch.save(torch.from_numpy(d[:, None]), os.path.join(with_d, "disparity.pt"))
    for instance in (with_d, without_d):
        os.makedirs(os.path.join(instance, "frames"))
        for i in range(9):  # 4 * (3 - 1) + 1 pixel frames
            open(os.path.join(instance, "frames", f"{i:05d}.png"), "wb").close()
    seen = []

    def fake_estimator(device):
        def estimate(paths, size):
            seen.append([os.path.basename(p) for p in paths])
            assert tuple(size) == d.shape[1:]
            return (d - 0.2) * 5.0

        return estimate

    monkeypatch.setattr(score_disparity, "DepthAnything", fake_estimator)
    monkeypatch.setattr(sys, "argv", ["score_disparity.py", "--runs-root", runs_root])
    score_disparity.main()
    score_disparity.main()  # already scored with this fit: skipped
    assert seen == [["00000.png", "00004.png", "00008.png"]]
    assert not os.path.exists(os.path.join(without_d, score_disparity.RESULT_FILE))
    with open(os.path.join(with_d, score_disparity.RESULT_FILE)) as f:
        result = json.load(f)
    assert result["fit"] == "affine" and result["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert [f["pixel_frame"] for f in result["frames"]] == [0, 4, 8]


def test_paired_difference_pairs_cases_and_is_reproducible():
    baseline = [1.0, 2.0, None, 4.0, 5.0]
    other = [1.5, 2.5, 9.0, None, 5.5]
    diff = report.paired_difference(baseline, other, resamples=500)
    assert diff["n"] == 3 and diff["mean"] == pytest.approx(0.5)
    assert diff["lo"] == pytest.approx(0.5) and diff["hi"] == pytest.approx(0.5)
    noisy = report.paired_difference([0.0] * 20, list(np.linspace(-1, 3, 20)), resamples=2000)
    assert noisy["lo"] < noisy["mean"] < noisy["hi"]
    assert noisy == report.paired_difference([0.0] * 20, list(np.linspace(-1, 3, 20)), 2000)
    assert report.paired_difference([None], [1.0], resamples=10) is None


def test_disparity_video_holds_each_latent_map_for_its_pixel_frames():
    assert report.latent_frame_per_pixel_frame(9).tolist() == [0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_case_groups_follow_summarize():
    cases = {
        "a": {"group": "pan_left"},
        "b": {"group": "push_in"},
        "c": {"group": "orbit_left"},
    }
    groups = report.case_groups(cases)
    assert list(groups) == [
        "push_in",
        "orbit_left",
        "pan_left",
        report.PARALLAX_GROUP,
        report.NO_PARALLAX_GROUP,
        report.ALL_GROUP,
    ]
    assert groups[report.PARALLAX_GROUP] == ["b", "c"] and groups[report.ALL_GROUP] == [
        "a",
        "b",
        "c",
    ]


def _report_instance(runs_root: str, rel: str, move: str, rot_err: float, num_frames: int = 9):
    from PIL import Image

    instance = os.path.join(runs_root, _common.OUTPUT_DIR, rel)
    os.makedirs(os.path.join(instance, "frames"))
    image_data = {
        "camera_path": [move],
        "scene_type": "indoor",
        "category": "Dining spaces",
        "prompt_list": ["image prompt", "a cozy cafe"],
    }
    with open(os.path.join(instance, "image_data.json"), "w") as f:
        json.dump(image_data, f)
    for i in range(num_frames):
        Image.new("RGB", (16, 8), (i, 0, 0)).save(os.path.join(instance, "frames", f"{i:05d}.png"))
    error = {"camera_control": {"camera_error": {"score": [rot_err, 0.1]}}}
    with open(os.path.join(instance, "evaluation.json"), "w") as f:
        json.dump(error, f)
    return instance


def test_report_pairs_runs_on_their_cases_and_links_videos(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    rels = {
        "push_in": os.path.join("static", "photorealistic", "indoor", "dining_spaces", "000"),
        "pan_left": os.path.join("static", "photorealistic", "indoor", "dining_spaces", "001"),
    }
    base, b2 = str(tmp_path / "base"), str(tmp_path / "b2")
    for move, rel in rels.items():
        _report_instance(base, rel, move, rot_err=2.0)
        instance = _report_instance(b2, rel, move, rot_err=1.5)
        d = torch.rand(3, 1, 4, 8)
        torch.save(d, os.path.join(instance, "disparity.pt"))
        with open(os.path.join(instance, "vbench.json"), "w") as f:
            json.dump({"imaging_quality": 0.6}, f)
    written = {}

    def fake_write_mp4(frames, path, fps=_common.FPS):
        written[os.path.basename(path)] = np.stack(list(frames))
        open(path, "wb").close()

    monkeypatch.setattr(report, "write_mp4", fake_write_mp4)
    out = str(tmp_path / "report")
    argv = ["report.py", "--out", out, "--run", f"base={base}", "--run", f"b2={b2}"]
    monkeypatch.setattr(sys, "argv", [*argv, "--bootstrap", "200"])
    report.main()

    push_id = _common.instance_id(rels["push_in"])
    assert written[f"{push_id}.mp4"].shape == (9, 8, 16, 3)
    assert written[f"{push_id}.disparity.mp4"].shape == (9, 4, 8, 3)
    with open(os.path.join(out, "report.json")) as f:
        result = json.load(f)
    rot = result["summary"]["push_in"]["rot err (deg)"]
    assert rot["diff"]["b2"]["mean"] == pytest.approx(-0.5) and rot["diff"]["b2"]["n"] == 1
    assert result["summary"][report.ALL_GROUP]["rot err (deg)"]["diff"]["b2"]["n"] == 2
    assert result["summary"][report.ALL_GROUP]["vbench imaging_quality"]["mean"]["base"] == [
        None,
        0,
    ]
    page = open(os.path.join(out, "index.html"), encoding="utf-8").read()
    assert f"media/b2/{push_id}.disparity.mp4" in page and "a cozy cafe" in page
    assert page.count("<section") == 2
