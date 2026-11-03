"""CPU tests for the SpatialVID disparity LMDB pass: real EXRs, a two-shard main LMDB."""

import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests.test_spatialvid_disparity import _ramp
from tests.test_spatialvid_lmdb import _FakeVAE, _make_clip
from tests.test_spatialvid_plan import _clip_id_with_late_start
from tools.data.wan21 import build_spatialvid_disparity_lmdb as bd
from tools.data.wan21.build_spatialvid_lmdb import (
    ClipDataset,
    make_loader,
    merge_shards,
    write_shard,
)
from tools.data.wan21.spatialvid import (
    DEFAULT_CAPTION_FIELD,
    load_annotation,
    plan_lmdb_clip,
)
from tools.data.wan21.spatialvid_disparity import (
    SKIP_INVALID_DEPTH,
    SKIP_INVERTED_DEPTH,
    DisparityStats,
    clip_disparity,
)

lmdb = pytest.importorskip("lmdb")
OpenEXR = pytest.importorskip("OpenEXR")
pytest.importorskip("av")
pytest.importorskip("cv2")


def _write_depth_zip(path, num_frames: int, near: float, nan: bool = False, flip: bool = False):
    """Depth zip with ``{i:05d}.exr`` per annotation frame; frame ``i`` is offset by ``i``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(num_frames):
            frame = (_ramp(near=near)[0] + np.float16(0.002 * i)).astype(np.float16)
            if flip:
                frame = frame[::-1].copy()
            if nan:
                frame[5, 5] = np.nan
            exr_path = path.parent / f"{i:05d}.exr"
            with OpenEXR.File(header, {"Z": frame}) as exr:
                exr.write(str(exr_path))
            archive.write(exr_path, f"{i:05d}.exr")
            exr_path.unlink()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Three clips in two groups, a two-shard main LMDB, and their depth zips."""
    root = tmp_path_factory.mktemp("spatialvid")
    late_id = _clip_id_with_late_start(np.arange(0, 241, 6), 24.0)
    specs = [  # id, fps, last frame, group, depth
        ("clip-a", 30, 144, 1, {"near": 0.6}),
        (late_id, 24, 240, 1, {"near": 0.9}),
        ("clip-nan", 30, 144, 2, {"near": 0.6, "nan": True}),
    ]
    rows = {}
    for clip_id, fps, last, group, depth in specs:
        rows[clip_id] = {**_make_clip(root, clip_id, fps, last, clip_id), "group id": str(group)}
        num_frames = len(load_annotation(str(root / "annotations" / clip_id)).frames)
        zip_path = root / "depths" / "depths" / f"group_{group:04d}" / f"{clip_id}.zip"
        _write_depth_zip(zip_path, num_frames, **depth)
    metadata = root / "metadata.csv"
    with open(metadata, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows["clip-a"]))
        writer.writeheader()
        writer.writerows(rows.values())

    # Rank 0 takes clip-a and clip-nan, rank 1 the late clip: the main LMDB order is
    # [clip-a, clip-nan, late], unlike the sorted ids [late, clip-a, clip-nan].
    ann_root, vid_root = str(root / "annotations"), str(root / "videos")
    shard_dirs = []
    for rank, ids in enumerate([["clip-a", "clip-nan"], [late_id]]):
        ds = ClipDataset([rows[c] for c in ids], ann_root, vid_root, DEFAULT_CAPTION_FIELD)
        env = lmdb.open(str(root / f".rank_{rank}"), map_size=1 << 28)
        write_shard(make_loader(ds, 0, pin_memory=False), env, _FakeVAE(), "cpu")
        env.close()
        shard_dirs.append(str(root / f".rank_{rank}"))
    merge_shards(shard_dirs, str(root / "main" / "data"))
    return {"root": root, "rows": rows, "late_id": late_id, "metadata": str(metadata)}


def _argv(dataset, output_dir, depths_root=None, num_workers=0) -> list[str]:
    root = dataset["root"]
    return [
        "--main_lmdb",
        str(root / "main" / "data"),
        "--metadata",
        dataset["metadata"],
        "--annotations_root",
        str(root / "annotations"),
        "--videos_root",
        str(root / "videos"),
        "--depths_root",
        str(depths_root or root / "depths"),
        "--output_dir",
        str(output_dir),
        "--num_workers",
        str(num_workers),
    ]


def test_depth_zip_path():
    row = {"id": "abc", "group id": "3"}
    assert bd.depth_zip_path(row, "/d") == "/d/depths/group_0003/abc.zip"


def test_pass_follows_main_lmdb_order_and_skips_nan_clip(dataset, tmp_path, capsys):
    root, late_id = dataset["root"], dataset["late_id"]
    assert sorted(dataset["rows"])[0] == late_id
    bd.main(_argv(dataset, tmp_path / "out", num_workers=2))
    out = capsys.readouterr().out

    env = lmdb.open(str(tmp_path / "out" / "data"), readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.get(b"disparity_shape") == b"2 20 1 240 416"
        assert txn.get(b"clip_ids_shape") == b"2"
        ids = [txn.get(f"clip_ids_{i}_data".encode()).decode() for i in range(2)]
        stored = [
            np.frombuffer(txn.get(f"disparity_{i}_data".encode()), np.float16).reshape(
                20, 1, 240, 416
            )
            for i in range(2)
        ]
    env.close()
    assert ids == ["clip-a", late_id]
    for clip_id, d in zip(ids, stored):
        row = dataset["rows"][clip_id]
        plan = plan_lmdb_clip(row, str(root / "annotations"), str(root / "videos"))
        expected, _ = clip_disparity(bd.depth_zip_path(row, str(root / "depths")), plan)
        np.testing.assert_array_equal(d, expected)
    assert not np.array_equal(stored[0], stored[1])

    skipped = (tmp_path / "out" / "skipped_clip_ids.txt").read_text()
    assert skipped == f"clip-nan {SKIP_INVALID_DEPTH}\n"
    assert (tmp_path / "out" / "failed_clip_ids.txt").read_text() == ""
    assert "2/3 clip-nan SKIP invalid_depth: 20 NaN/inf depth pixels" in out
    assert f"written 2, skipped 1 ({SKIP_INVALID_DEPTH}=1), failed 0 of 3" in out
    assert "std(2d-1)=" in out and "1/std=" in out


def test_inverted_depth_stops_the_pass(dataset, tmp_path):
    root = dataset["root"]
    depths = tmp_path / "depths"
    for clip_id, row in dataset["rows"].items():
        num_frames = len(load_annotation(str(root / "annotations" / clip_id)).frames)
        zip_path = Path(bd.depth_zip_path(row, str(depths)))
        _write_depth_zip(zip_path, num_frames, near=0.6, flip=clip_id == "clip-a")
    with pytest.raises(RuntimeError, match="group 1: 1 of 2 clips skipped as inverted_depth"):
        bd.main(_argv(dataset, tmp_path / "out", depths_root=depths))
    env = lmdb.open(str(tmp_path / "out" / "data"), readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.get(b"disparity_shape") is None
    env.close()
    skipped = (tmp_path / "out" / "skipped_clip_ids.txt").read_text()
    assert skipped == f"clip-a {SKIP_INVERTED_DEPTH}\n"


def test_plan_that_differs_from_main_lmdb_fails(dataset):
    root = dataset["root"]
    sample = bd.read_main_lmdb(str(root / "main" / "data"))[0]
    assert sample["clip_ids"] == "clip-a"
    moved = {**sample, "poses": sample["poses"] + np.float32(1e-3)}
    roots = [str(root / d) for d in ("annotations", "videos", "depths")]
    ok = bd.label_clip((sample, dataset["rows"]["clip-a"]), *roots)
    assert ok["status"] == "ok"
    bad = bd.label_clip((moved, dataset["rows"]["clip-a"]), *roots)
    assert bad["status"] == "error" and "differs from the main LMDB" in bad["message"]


# --- tally -------------------------------------------------------------------------------------


def test_stop_rule_allows_five_percent_per_group():
    groups = {f"g1-{i}": "1" for i in range(40)} | {f"g2-{i}": "2" for i in range(40)}
    tally = bd.DisparityTally(groups)
    tally.add_skip("g1-0", SKIP_INVERTED_DEPTH)
    tally.add_skip("g1-1", SKIP_INVERTED_DEPTH)
    tally.add_skip("g2-0", SKIP_INVERTED_DEPTH)
    for i in range(10):
        tally.add_skip(f"g2-{i + 1}", SKIP_INVALID_DEPTH)
    with pytest.raises(RuntimeError, match="group 1: 3 of 40"):
        tally.add_skip("g1-2", SKIP_INVERTED_DEPTH)


def test_tally_moments_and_summary(tmp_path):
    rng = np.random.default_rng(0)
    labels = [rng.uniform(0, 1, (20, 1, 240, 416)).astype(np.float16) for _ in range(3)]
    tally = bd.DisparityTally({"a": "1", "b": "1", "c": "1", "d": "1", "e": "1"})
    for d, frac in zip(labels, [0.1, 0.3, 0.2]):
        tally.add_label(d, frac)
    tally.add_skip("d", SKIP_INVALID_DEPTH)
    tally.failed.append("e")
    all_x = 2 * np.concatenate([d.astype(np.float64).ravel() for d in labels]) - 1
    assert tally.std() == pytest.approx(np.std(all_x), rel=1e-6)
    lines = tally.summary()
    assert lines[0] == "written 3, skipped 1 (invalid_depth=1), failed 1 of 5"
    assert lines[1] == "clipped_fraction median=0.2000 P90=0.3000"
    std = np.std(all_x)
    assert lines[2].startswith(f"std(2d-1)={std:.4f} 1/std={1 / std:.4f} a (1/std to 0.5)=")
    assert lines[2].endswith("=1.5")  # uniform d: std(2d−1) = 1/√3, 1/std = 1.73
    tally.write_lists(str(tmp_path))
    assert (tmp_path / "skipped_clip_ids.txt").read_text() == "d invalid_depth\n"
    assert (tmp_path / "failed_clip_ids.txt").read_text() == "e\n"


def test_writer_drops_a_malformed_label(tmp_path):
    tally = bd.DisparityTally({"a": "1", "b": "1"})
    stats = DisparityStats(p90=0.5, clipped_fraction=0.0, near_below_fraction=1.0)
    results = [
        {"status": "ok", "clip_ids": "a", "seconds": 0.0, "stats": stats, "disparity": d}
        for d in (np.zeros((20, 1, 240, 416), np.float32), np.zeros((20, 1, 240, 416), np.float16))
    ]
    results[1]["clip_ids"] = "b"
    env = lmdb.open(str(tmp_path / "data"), map_size=1 << 26)
    bd.write_disparity_lmdb(results, env, tally)
    with env.begin() as txn:
        assert txn.get(b"clip_ids_shape") == b"1"
        assert txn.get(b"clip_ids_0_data") == b"b"
    env.close()
    assert tally.failed == ["a"] and tally.written == 1
