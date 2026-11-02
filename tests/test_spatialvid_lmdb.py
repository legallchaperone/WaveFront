"""CPU tests for SpatialVID frame decoding and the LMDB writer, with a stand-in VAE."""

import csv
import json

import numpy as np
import pytest
import torch

from minwm.data.datasets.lmdb import CameraLatentLMDBDataset
from tests.test_spatialvid_plan import _annotation, _clip_id_with_late_start, _forward_trajectory
from tools.data.wan21.build_spatialvid_lmdb import (
    LATENT_SHAPE,
    ClipDataset,
    frames_to_pixels,
    make_loader,
    merge_shards,
    prepare_clip,
    write_shard,
)
from tools.data.wan21.spatialvid import (
    CAPTION_FIELDS,
    DEFAULT_CAPTION_FIELD,
    crop_resize_frame,
    decode_clip_frames,
    load_annotation,
    load_caption,
    main,
    output_source_frames,
    plan_clip,
    video_is_hdr,
)

av = pytest.importorskip("av")
lmdb = pytest.importorskip("lmdb")
pytest.importorskip("cv2")

NUM_BITS = 9  # frame indices up to 511
STRIPE = 1248 // NUM_BITS  # source columns per bit inside the crop window


def _coded_frame(index: int) -> np.ndarray:
    """720×1280 RGB frame whose crop window carries ``index`` as black/white stripes."""
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :16] = img[:, 1264:] = 128  # margins, cropped away
    for bit in range(NUM_BITS):
        if index >> bit & 1:
            img[:, 16 + bit * STRIPE : 16 + (bit + 1) * STRIPE] = 255
    return img


def _read_code(frame: np.ndarray) -> int:
    """Index carried by a ``(480, 832, 3)`` uint8 frame after crop and resize."""
    columns = [int((bit + 0.5) * STRIPE / 1.5) for bit in range(NUM_BITS)]
    return int(sum(1 << bit for bit, x in enumerate(columns) if frame[:, x].mean() > 127))


def _write_video(path, num_frames: int, fps: int) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps, options={"bf": "2", "qscale": "2"})
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        for i in range(num_frames):
            frame = av.VideoFrame.from_ndarray(_coded_frame(i), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# --- crop and resize ---------------------------------------------------------------------------


def test_crop_resize_drops_margins_and_area_averages():
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[:, :16] = frame[:, 1264:] = 250
    # A 3×3 block at cropped (row 150, col 300) is exactly output pixels rows 100–101, cols
    # 200–201 under the 1.5× area resize.
    frame[150:153, 16 + 300 : 16 + 303] = 240
    out = crop_resize_frame(frame)
    assert out.shape == (480, 832, 3) and out.dtype == np.uint8
    block = np.zeros((480, 832), bool)
    block[100:102, 200:202] = True
    assert np.all(out[block] == 240)
    assert np.all(out[~block] == 60)

    # A 1-pixel column at cropped col 301 covers a third of output cols 200 and 201 each:
    # area averaging gives 60 + 180/3 = 120 there (bilinear would give 105).
    line = np.full((720, 1280, 3), 60, np.uint8)
    line[:, 16 + 301] = 240
    out = crop_resize_frame(line)
    assert np.all(out[:, 200:202] == 120)
    assert np.all(out[:, :200] == 60) and np.all(out[:, 202:] == 60)


def test_crop_resize_rejects_other_sizes():
    with pytest.raises(ValueError, match="1280x720"):
        crop_resize_frame(np.zeros((1080, 1920, 3), np.uint8))


def test_frames_to_pixels():
    frames = torch.tensor([0, 255, 51], dtype=torch.uint8).view(3, 1, 1, 1).expand(3, 2, 2, 3)
    pixels = frames_to_pixels(frames)
    assert pixels.shape == (3, 3, 2, 2) and pixels.dtype == torch.float32
    np.testing.assert_allclose(pixels[:, :, 0, 0].numpy(), [[-1, 1, 51 / 127.5 - 1]] * 3)


# --- frame decoding ----------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0, 150])
def test_decode_clip_frames_returns_exact_source_frames(tmp_path, start):
    video = tmp_path / "clip.mp4"
    _write_video(video, num_frames=start + 146, fps=30)
    source_frames = output_source_frames(30.0, start)
    frames = decode_clip_frames(str(video), source_frames)
    assert frames.shape == (77, 480, 832, 3) and frames.dtype == np.uint8
    assert [_read_code(f) for f in frames] == source_frames.tolist()


def test_decode_clip_frames_repeats_and_reports_short_video(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video, num_frames=12, fps=30)
    frames = decode_clip_frames(str(video), np.array([0, 3, 3, 11]))
    assert [_read_code(f) for f in frames] == [0, 3, 3, 11]
    with pytest.raises(ValueError, match="12 frames decoded, need frame 20"):
        decode_clip_frames(str(video), np.array([0, 20]))


# --- HDR detection -----------------------------------------------------------------------------


def _write_small(path, codec: str, pix_fmt: str, options: dict | None = None) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=30, options=options or {})
        stream.width, stream.height, stream.pix_fmt = 64, 48, pix_fmt
        rgb = np.zeros((48, 64, 3), np.uint8)
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format=pix_fmt)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize(
    "codec, pix_fmt, options, hdr",
    [
        ("mpeg4", "yuv420p", None, False),
        ("libx264", "yuv420p", {"color_trc": "bt709"}, False),
        ("libx264", "yuv420p", {"color_trc": "arib-std-b67"}, True),
        ("libx264", "yuv420p", {"color_trc": "smpte2084"}, True),
        ("libx264", "yuv420p10le", None, True),
    ],
)
def test_video_is_hdr(tmp_path, codec, pix_fmt, options, hdr):
    if codec not in av.codecs_available:
        pytest.skip(f"{codec} not in this PyAV build")
    path = tmp_path / "v.mp4"
    try:
        _write_small(path, codec, pix_fmt, options)
    except av.error.FFmpegError as e:
        pytest.skip(f"{codec} cannot write {pix_fmt} here: {e}")
    assert video_is_hdr(str(path)) is hdr


# --- captions ----------------------------------------------------------------------------------


def test_load_caption(tmp_path):
    assert DEFAULT_CAPTION_FIELD == "SceneDescription"
    (tmp_path / "caption.json").write_text(
        json.dumps({"SceneDescription": "  A street.  ", "CameraMotion": ""})
    )
    assert load_caption(str(tmp_path), "SceneDescription") == "A street."
    with pytest.raises(ValueError, match="no CameraMotion"):
        load_caption(str(tmp_path), "CameraMotion")
    with pytest.raises(ValueError, match="not one of"):
        load_caption(str(tmp_path), "CategoryTags")


# --- LMDB writer -------------------------------------------------------------------------------


class _FakeVAE:
    """Stand-in for Wan21VAE: latent frame k carries the code of pixel frame 4k."""

    def __init__(self):
        self.inputs = []

    def encode(self, videos):
        out = []
        for video in videos:
            self.inputs.append((tuple(video.shape), video.dtype, float(video.min())))
            frames = ((video.permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
            codes = [_read_code(frames[4 * k]) for k in range(LATENT_SHAPE[0])]
            latent = torch.tensor(codes, dtype=torch.float32)[None, :, None, None]
            out.append(latent.expand(16, LATENT_SHAPE[0], 60, 104).clone())
        return out


def _make_clip(root, clip_id: str, fps: int, last_frame: int, caption: str) -> dict[str, str]:
    frames = np.arange(0, last_frame + 1, 6)
    ann = _annotation(
        _forward_trajectory(frames, fps, speed=0.4, yaw_deg_per_s=10.0), frames, clip_id=clip_id
    )
    ann_dir = root / "annotations" / clip_id
    ann_dir.mkdir(parents=True)
    lines = [f"# total {len(frames)} indexes"] + [f"{i} {f}" for i, f in enumerate(frames)]
    (ann_dir / "indexes.txt").write_text("\n".join(lines) + "\n")
    np.save(ann_dir / "poses.npy", ann.poses.astype(np.float32))
    np.save(ann_dir / "intrinsics.npy", ann.intrinsics.astype(np.float32))
    (ann_dir / "caption.json").write_text(json.dumps({f: f"{caption} {f}" for f in CAPTION_FIELDS}))
    (root / "videos").mkdir(exist_ok=True)
    _write_video(root / "videos" / f"{clip_id}.mp4", num_frames=last_frame + 1, fps=fps)
    return {
        "id": clip_id,
        "fps": str(fps),
        "resolution": "1280x720",
        "dynamicRatio": "0.1",
        "annotation path": clip_id,
        "video path": f"{clip_id}.mp4",
    }


def test_lmdb_round_trip_through_camera_dataset(tmp_path):
    late_id = _clip_id_with_late_start(np.arange(0, 241, 6), 24.0)
    rows = [
        _make_clip(tmp_path, "clip-a", fps=30, last_frame=144, caption="a"),
        _make_clip(tmp_path, late_id, fps=24, last_frame=240, caption="b"),
        _make_clip(tmp_path, "clip-short", fps=30, last_frame=120, caption="s"),
    ]
    ann_root, vid_root = str(tmp_path / "annotations"), str(tmp_path / "videos")
    plans = [
        plan_clip(
            load_annotation(f"{ann_root}/{row['id']}"),
            row,
            hdr=False,
            min_window_path=0.0,
            max_dynamic_ratio=1.0,
        )
        for row in rows[:2]
    ]
    assert plans[0].start_frame == 0 and plans[1].start_frame > 0

    # Two ranks: rank 0 gets clip-a and the short clip (decoded by worker processes), rank 1
    # gets the late-start clip (decoded in the main process).
    vae = _FakeVAE()
    shard_dirs, results = [], []
    for rank, (shard_rows, workers) in enumerate([([rows[0], rows[2]], 2), ([rows[1]], 0)]):
        dataset = ClipDataset(shard_rows, ann_root, vid_root, DEFAULT_CAPTION_FIELD)
        env = lmdb.open(str(tmp_path / f".rank_{rank}"), map_size=1 << 28)
        results.append(
            write_shard(make_loader(dataset, workers, pin_memory=False), env, vae, "cpu")
        )
        env.close()
        shard_dirs.append(str(tmp_path / f".rank_{rank}"))
    assert results == [(1, ["clip-short"]), (1, [])]
    assert vae.inputs == [((3, 77, 480, 832), torch.float32, -1.0)] * 2

    out = tmp_path / "out" / "data"
    assert merge_shards(shard_dirs, str(out)) == (2, ["clip-short"])
    env = lmdb.open(str(out), readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.get(b"latents_shape") == b"2 20 16 60 104"
        assert txn.get(b"poses_shape") == b"2 20 7"
        assert txn.get(b"intrinsics_shape") == b"2 4"
        assert txn.get(b"prompts_shape") == b"2"
        stored_ids = [txn.get(f"clip_ids_{i}_data".encode()).decode() for i in range(2)]
    env.close()
    assert stored_ids == ["clip-a", late_id]

    ds = CameraLatentLMDBDataset(str(out))
    assert len(ds) == 2
    for i, (row, plan, caption) in enumerate(zip(rows, plans, "ab")):
        item = ds[i]
        latent = item["clean_latent"]
        assert latent.shape == (20, 16, 60, 104) and latent.dtype == torch.float32
        np.testing.assert_array_equal(latent[:, 0, 0, 0].numpy(), plan.source_frames[::4])
        assert item["prompts"] == f"{caption} SceneDescription"
        assert item["viewmats"].shape == (20, 4, 4) and item["Ks"].shape == (20, 3, 3)
        np.testing.assert_allclose(item["viewmats"][0].numpy(), np.eye(4), atol=1e-6)
        fx, fy, cx, cy = plan.intrinsics
        np.testing.assert_allclose(
            item["Ks"][0].numpy(), [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], rtol=1e-6
        )
        np.testing.assert_allclose(
            item["viewmats"].numpy(), _viewmats(plan.poses), atol=1e-5, err_msg=row["id"]
        )


def _viewmats(poses: np.ndarray) -> np.ndarray:
    from minwm.data.datasets.geometry import build_viewmats_and_Ks

    return build_viewmats_and_Ks(np.array([0.5, 0.9, 0.5, 0.5]), poses)[0]


def test_prepare_clip_rejects_contract_skips(tmp_path):
    row = _make_clip(tmp_path, "clip-s", fps=60, last_frame=120, caption="s")
    with pytest.raises(ValueError, match="short_span"):
        prepare_clip(row, str(tmp_path / "annotations"), str(tmp_path / "videos"), "SceneSummary")


# --- planner CLI -------------------------------------------------------------------------------


def test_planner_cli_reports_bad_annotations_as_errors(tmp_path, capsys):
    rows = [
        _make_clip(tmp_path, "clip-good", fps=30, last_frame=144, caption="g"),
        _make_clip(tmp_path, "clip-short", fps=30, last_frame=120, caption="s"),
        _make_clip(tmp_path, "clip-bad", fps=30, last_frame=144, caption="b"),
    ]
    # indexes.txt keeps 25 rows while poses.npy loses some: which rows line up is unknown.
    bad_poses = tmp_path / "annotations" / "clip-bad" / "poses.npy"
    np.save(bad_poses, np.load(bad_poses)[:20])
    metadata = tmp_path / "metadata.csv"
    with open(metadata, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "group id"])
        writer.writeheader()
        writer.writerows({**row, "group id": "7"} for row in rows)
    output = tmp_path / "kept.txt"

    main(
        [
            "--metadata",
            str(metadata),
            "--annotations_root",
            str(tmp_path / "annotations"),
            "--videos_root",
            str(tmp_path / "videos"),
            "--group",
            "7",
            "--output",
            str(output),
        ]
    )
    out = capsys.readouterr().out
    assert "clip-bad ERROR" in out and "25 indexes, 20 poses, 25 intrinsics" in out
    assert "clip-short" in out and "SKIP short_span" in out
    assert out.strip().splitlines()[-1] == "kept 1, skipped 1, errors 1 of 3"
    assert output.read_text() == "clip-good\n"
