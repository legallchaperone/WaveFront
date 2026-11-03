"""CPU tests for CameraDisparityLMDBDataset: a main LMDB joined with its disparity LMDB."""

import numpy as np
import pytest
import torch

from minwm.config import build

lmdb = pytest.importorskip("lmdb")
pytest.importorskip("scipy")

from minwm.data.datasets.lmdb import CameraDisparityLMDBDataset, CameraLatentLMDBDataset
from minwm.data.datasets.mock import MockCameraLatentDataset

F, C, H, W = 4, 16, 4, 4
DISP = (F, 1, 4 * H, 4 * W)
MAIN_IDS = ["clip-c", "clip-a", "clip-d", "clip-b", "clip-e"]


def _write(path, samples: list[dict], shapes: dict[str, tuple]) -> str:
    """Write ``samples`` in the minWM LMDB layout: ``<key>_shape`` plus ``<key>_<i>_data``."""
    env = lmdb.open(str(path), map_size=1 << 26)
    with env.begin(write=True) as txn:
        for i, sample in enumerate(samples):
            for key, value in sample.items():
                raw = value.encode() if isinstance(value, str) else value.tobytes()
                txn.put(f"{key}_{i}_data".encode(), raw)
        for key, shape in shapes.items():
            txn.put(f"{key}_shape".encode(), " ".join(map(str, (len(samples), *shape))).encode())
    env.close()
    return str(path)


def _main_lmdb(path, ids=MAIN_IDS) -> str:
    samples = [
        {
            "latents": np.full((F, C, H, W), i, np.float16),
            "prompts": f"prompt {clip_id}",
            "intrinsics": np.array([0.5, 0.9, 0.5, 0.5], np.float32),
            "poses": np.tile(np.array([0, 0, 0.1 * i, 0, 0, 0, 1], np.float32), (F, 1)),
            "clip_ids": clip_id,
        }
        for i, clip_id in enumerate(ids)
    ]
    shapes = {"latents": (F, C, H, W), "intrinsics": (4,), "poses": (F, 7)}
    return _write(path, samples, {**shapes, "prompts": (), "clip_ids": ()})


def _disparity_lmdb(path, ids, shape=DISP) -> tuple[str, list[np.ndarray]]:
    """The disparity LMDB and its labels (lmdb refuses a second handle on an open LMDB)."""
    rng = np.random.default_rng(0)
    labels = [rng.uniform(0, 1, shape).astype(np.float16) for _ in ids]
    samples = [{"disparity": d, "clip_ids": clip_id} for d, clip_id in zip(labels, ids)]
    return _write(path, samples, {"disparity": shape, "clip_ids": ()}), labels


def test_joins_labelled_clips_in_main_order(tmp_path):
    main = _main_lmdb(tmp_path / "main")
    # clip-a, clip-b and clip-e have labels: main samples 1, 3, 4, not in sorted-id order of
    # the main LMDB (clip-a < clip-b < clip-c ...).
    disp, labels = _disparity_lmdb(tmp_path / "disp", ["clip-a", "clip-b", "clip-e"])
    ds = CameraDisparityLMDBDataset(main, disp)
    plain = CameraLatentLMDBDataset(_main_lmdb(tmp_path / "main_copy"))
    assert len(ds) == 3
    for i, (row, clip_id) in enumerate(zip([1, 3, 4], ["clip-a", "clip-b", "clip-e"])):
        item, ref = ds[i], plain[row]
        assert item["prompts"] == f"prompt {clip_id}"
        assert set(item) == set(ref) | {"disparity"}
        for key in ("clean_latent", "viewmats", "Ks"):
            torch.testing.assert_close(item[key], ref[key], rtol=0, atol=0)
        torch.testing.assert_close(
            item["disparity"], torch.from_numpy(labels[i].astype(np.float32))
        )
    capped = CameraDisparityLMDBDataset(
        _main_lmdb(tmp_path / "main_2"),
        _disparity_lmdb(tmp_path / "disp_2", ["clip-a", "clip-b", "clip-e"])[0],
        max_pair=2,
    )
    assert len(capped) == 2


def test_sample_matches_the_mock_contract(tmp_path):
    ds = CameraDisparityLMDBDataset(
        _main_lmdb(tmp_path / "main"), _disparity_lmdb(tmp_path / "disp", ["clip-d"])[0]
    )
    mock = MockCameraLatentDataset(
        length=1, num_frames=F, channels=C, height=H, width=W, with_disparity=True
    )[0]
    item = ds[0]
    assert set(item) == set(mock)
    for key, value in mock.items():
        if isinstance(value, torch.Tensor):
            assert item[key].shape == value.shape and item[key].dtype == value.dtype, key
    assert 0 <= float(item["disparity"].min()) and float(item["disparity"].max()) <= 1


def test_builds_from_a_config_node(tmp_path):
    main = _main_lmdb(tmp_path / "main")
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-c"])
    ds = build(dict(type="CameraDisparityLMDBDataset", data_path=main, disparity_path=disp))
    assert isinstance(ds, CameraDisparityLMDBDataset) and len(ds) == 1


def test_missing_clip_raises(tmp_path):
    main = _main_lmdb(tmp_path / "main")
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-a", "clip-x"])
    with pytest.raises(ValueError, match="clip clip-x \\(label 1\\) not in"):
        CameraDisparityLMDBDataset(main, disp)


def test_out_of_order_raises(tmp_path):
    main = _main_lmdb(tmp_path / "main")
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-b", "clip-a"])
    with pytest.raises(ValueError, match="clip clip-a \\(label 1\\) is main sample 1, not after 3"):
        CameraDisparityLMDBDataset(main, disp)


def test_repeated_main_clip_raises(tmp_path):
    main = _main_lmdb(tmp_path / "main", ids=["clip-a", "clip-b", "clip-a"])
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-a"])
    with pytest.raises(ValueError, match="clip clip-a at samples 0, 2"):
        CameraDisparityLMDBDataset(main, disp)


def test_disparity_shape_must_match_the_latent_grid(tmp_path):
    main = _main_lmdb(tmp_path / "main")
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-a"], shape=(F, 1, 2 * H, 2 * W))
    with pytest.raises(ValueError, match="want \\(4, 1, 16, 16\\)"):
        CameraDisparityLMDBDataset(main, disp)


def test_sharded_main_lmdb_is_rejected(tmp_path):
    (tmp_path / "shards").mkdir()
    _main_lmdb(tmp_path / "shards" / "0")
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-a"])
    with pytest.raises(ValueError, match="single LMDB"):
        CameraDisparityLMDBDataset(str(tmp_path / "shards"), disp)


def test_item_clip_id_mismatch_raises(tmp_path):
    main = _main_lmdb(tmp_path / "main")
    disp, _ = _disparity_lmdb(tmp_path / "disp", ["clip-a", "clip-b"])
    ds = CameraDisparityLMDBDataset(main, disp)
    ds[1]
    ds._main_rows[1] = 4  # a stale index pointing at another clip
    with pytest.raises(RuntimeError, match="label 1 is clip clip-b, main sample 4 is clip-e"):
        ds[1]


def test_dataloader_with_two_workers(tmp_path):
    from torch.utils.data import DataLoader

    main = _main_lmdb(tmp_path / "main")
    ids = ["clip-c", "clip-d", "clip-b", "clip-e"]
    disp, labels = _disparity_lmdb(tmp_path / "disp", ids)
    ds = CameraDisparityLMDBDataset(main, disp)
    # The LMDB handles are opened at construction, like CameraLatentLMDBDataset's, so workers
    # must fork (the Linux default; macOS spawns, which cannot pickle them).
    loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context="fork")
    batches = list(loader)
    assert [b["prompts"] for b in batches] == [
        ["prompt clip-c", "prompt clip-d"],
        ["prompt clip-b", "prompt clip-e"],
    ]
    assert batches[0]["disparity"].shape == (2, *DISP)
    assert batches[0]["disparity"].dtype == torch.float32
    torch.testing.assert_close(
        batches[1]["disparity"][1], torch.from_numpy(labels[3].astype(np.float32))
    )
