"""Encode planned SpatialVID-HQ clips into the camera-aware LMDB that WaveFront trains on.

Takes the clip ids that ``spatialvid.py`` kept and writes, per clip, the keys of
``docs/ai-generated/decisions.md`` (section "2026-11-01：SpatialVID 数据格式"):

    latents    (20, 16, 60, 104) float16  Wan2.1 VAE latent of the 77 planned pixel frames
    prompts    str                        ``caption.json`` ``SceneDescription`` by default
    intrinsics (4,) float32               [fx/W, fy/H, cx/W, cy/H] after crop and resize
    poses      (20, 7) float32            w2c OpenCV [tx, ty, tz, qx, qy, qz, qw]
    clip_ids   str                        SpatialVID clip id, to join later passes one-to-one

The layout matches ``build_worldplaygen_lmdb.py``: ``<key>_shape`` plus ``<key>_<i>_data``. Each
rank streams into its own shard, then rank 0 merges the shards into ``<output_dir>/data``.

Per rank, DataLoader workers plan, decode, crop and resize clips, the main process runs the VAE,
and a writer thread commits each sample to the LMDB while the next clip is encoded. Each clip
logs how long the main process waited for it, how long a worker took to decode it, and how long
the encode and the write took.

Usage (from the repo root, as a module so ``tools.data.wan21.spatialvid`` resolves)::

    torchrun --nproc_per_node=4 -m tools.data.wan21.build_spatialvid_lmdb \\
        --clip_ids clip_ids_group_0001.txt \\
        --metadata /path/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv \\
        --annotations_root /path/SpatialVID-HQ/extracted/annotations/SpatialVID \\
        --videos_root /path/SpatialVID-HQ/extracted/videos/SpatialVID \\
        --output_dir /path/spatialvid/train \\
        --vae_path wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
"""

import argparse
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.data.wan21.spatialvid import (
    CAPTION_FIELDS,
    DEFAULT_CAPTION_FIELD,
    NUM_LATENT_FRAMES,
    decode_clip_frames,
    load_caption,
    plan_lmdb_clip,
    read_metadata,
)

LATENT_SHAPE = (NUM_LATENT_FRAMES, 16, 60, 104)
ARRAY_KEYS = {
    "latents": (np.float16, LATENT_SHAPE),
    "intrinsics": (np.float32, (4,)),
    "poses": (np.float32, (NUM_LATENT_FRAMES, 7)),
}
STRING_KEYS = ("prompts", "clip_ids")
PER_SAMPLE_BYTES = int(np.prod(LATENT_SHAPE)) * 2 + 4 * 4 + NUM_LATENT_FRAMES * 7 * 4 + 4000


def prepare_clip(
    row: Mapping, annotations_root: str, videos_root: str, caption_field: str
) -> dict[str, Any]:
    """Plan one clip and decode its frames: the CPU half of one LMDB sample.

    Args:
        row (Mapping): the clip's metadata row.
        annotations_root (str): directory the ``annotation path`` column is relative to.
        videos_root (str): directory the ``video path`` column is relative to.
        caption_field (str): which ``caption.json`` field becomes the prompt.

    Returns:
        dict[str, Any]: ``frames`` ``(77, 480, 832, 3)`` uint8, ``intrinsics``, ``poses``,
        ``prompts``, ``clip_ids`` and ``decode_s`` (seconds spent decoding).

    Raises:
        ValueError: if a contract rule skips the clip (see
            :func:`~tools.data.wan21.spatialvid.plan_lmdb_clip`).
    """
    ann_dir = os.path.join(annotations_root, row["annotation path"])
    video_path = os.path.join(videos_root, row["video path"])
    plan = plan_lmdb_clip(row, annotations_root, videos_root)
    t0 = time.perf_counter()
    frames = decode_clip_frames(video_path, plan.source_frames)
    return {
        "frames": frames,
        "intrinsics": plan.intrinsics,
        "poses": plan.poses,
        "prompts": load_caption(ann_dir, caption_field),
        "clip_ids": plan.clip_id,
        "decode_s": time.perf_counter() - t0,
    }


class ClipDataset(Dataset):
    """Runs :func:`prepare_clip` in DataLoader workers; a failure becomes an ``error`` item.

    Args:
        rows (Sequence[Mapping]): metadata rows of the clips, in output order.
        annotations_root (str): directory the ``annotation path`` column is relative to.
        videos_root (str): directory the ``video path`` column is relative to.
        caption_field (str): which ``caption.json`` field becomes the prompt.
    """

    def __init__(
        self, rows: Sequence[Mapping], annotations_root: str, videos_root: str, caption_field: str
    ):
        self.rows = list(rows)
        self.annotations_root = annotations_root
        self.videos_root = videos_root
        self.caption_field = caption_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        try:
            item = prepare_clip(row, self.annotations_root, self.videos_root, self.caption_field)
        except Exception as e:
            return {"clip_ids": row["id"], "error": repr(e)}
        item["error"] = ""
        return item


def frames_to_pixels(frames: torch.Tensor) -> torch.Tensor:
    """``(T, H, W, 3)`` uint8 frames to the VAE input ``(3, T, H, W)`` float32 in ``[-1, 1]``."""
    return frames.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def encode_item(item: Mapping[str, Any], vae: Any, device: torch.device | str) -> dict[str, Any]:
    """Encode the frames of a :func:`prepare_clip` item into one LMDB sample.

    Args:
        item (Mapping[str, Any]): a :class:`ClipDataset` item without an error.
        vae (Any): object with ``encode(list[Tensor[3, T, H, W]]) -> list[Tensor[16, F, h, w]]``,
            e.g. :class:`minwm.modeling.wan21.vae.Wan21VAE`.
        device (torch.device | str): device the VAE runs on.

    Returns:
        dict[str, Any]: one value per LMDB key.

    Raises:
        ValueError: if a value has the wrong shape or dtype.
    """
    frames = torch.as_tensor(item["frames"]).to(device, non_blocking=True)
    with torch.no_grad():
        latent = vae.encode([frames_to_pixels(frames)])[0]
    sample = {
        "latents": latent.permute(1, 0, 2, 3).to(torch.float16).cpu().numpy(),
        "intrinsics": np.asarray(item["intrinsics"]),
        "poses": np.asarray(item["poses"]),
        "prompts": item["prompts"],
        "clip_ids": item["clip_ids"],
    }
    for key, (dtype, shape) in ARRAY_KEYS.items():
        if sample[key].shape != shape or sample[key].dtype != dtype:
            raise ValueError(
                f"{item['clip_ids']}: {key} is {sample[key].dtype}{sample[key].shape}, "
                f"want {np.dtype(dtype)}{shape}"
            )
    return sample


def put_sample(txn: Any, index: int, sample: Mapping[str, Any]) -> None:
    """Write one sample as ``<key>_<index>_data`` entries in an open LMDB write transaction.

    Args:
        txn (Any): an ``lmdb`` write transaction.
        index (int): sample index within this LMDB.
        sample (Mapping[str, Any]): the values from :func:`encode_item`.
    """
    for key in ARRAY_KEYS:
        txn.put(f"{key}_{index}_data".encode(), np.ascontiguousarray(sample[key]).tobytes())
    for key in STRING_KEYS:
        txn.put(f"{key}_{index}_data".encode(), sample[key].encode())


def write_shard(
    loader: Any, env: Any, vae: Any, device: torch.device | str, log_prefix: str = ""
) -> tuple[int, list[str]]:
    """Encode every item of ``loader`` into ``env``, logging per-clip timing.

    A writer thread commits each sample while the next clip is encoded, so the GPU does not
    idle during the LMDB write. Samples keep the loader's order.

    Args:
        loader (Any): iterable of :class:`ClipDataset` items, e.g. a DataLoader.
        env (Any): the ``lmdb`` environment of this rank's shard.
        vae (Any): the VAE, as in :func:`encode_item`.
        device (torch.device | str): device the VAE runs on.
        log_prefix (str): prefix of every log line, e.g. ``"rank 0: "``.

    Returns:
        tuple[int, list[str]]: samples written and the clip ids that failed. ``__count__`` and
        ``__failed__`` are also stored in ``env``.
    """
    import queue
    import threading

    pending: queue.Queue = queue.Queue(maxsize=2)
    written, failed = [0], []

    def _writer() -> None:
        while (job := pending.get()) is not None:
            sample, timing = job
            clip_id = sample["clip_ids"]
            t0 = time.perf_counter()
            try:
                with env.begin(write=True) as txn:
                    put_sample(txn, written[0], sample)
            except Exception as e:
                print(f"{log_prefix}{clip_id} failed: {e!r}", flush=True)
                failed.append(clip_id)
                continue
            written[0] += 1
            print(f"{timing} write={time.perf_counter() - t0:.2f}s", flush=True)

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    total = len(loader) if hasattr(loader, "__len__") else "?"
    t_wait = time.perf_counter()
    try:
        for i, item in enumerate(loader):
            wait = time.perf_counter() - t_wait
            clip_id = item["clip_ids"]
            if item["error"]:
                print(f"{log_prefix}{clip_id} failed: {item['error']}", flush=True)
                failed.append(clip_id)
                t_wait = time.perf_counter()
                continue
            try:
                t0 = time.perf_counter()
                sample = encode_item(item, vae, device)
                t1 = time.perf_counter()
            except Exception as e:
                print(f"{log_prefix}{clip_id} failed: {e!r}", flush=True)
                failed.append(clip_id)
                t_wait = time.perf_counter()
                continue
            timing = (
                f"{log_prefix}{i + 1}/{total} {clip_id} wait={wait:.2f}s "
                f"decode={float(item['decode_s']):.2f}s encode={t1 - t0:.2f}s"
            )
            pending.put((sample, timing))
            t_wait = time.perf_counter()
    finally:
        pending.put(None)
        writer.join()
    count = written[0]
    with env.begin(write=True) as txn:
        txn.put(b"__count__", str(count).encode())
        txn.put(b"__failed__", "\n".join(failed).encode())
    return count, failed


def merge_shards(shard_dirs: list[str], out_dir: str) -> tuple[int, list[str]]:
    """Concatenate per-rank shards into one LMDB and write the ``<key>_shape`` entries.

    Args:
        shard_dirs (list[str]): per-rank LMDB directories, in rank order; each holds
            ``__count__``, ``__failed__`` and samples ``0 .. count-1``.
        out_dir (str): directory of the merged LMDB.

    Returns:
        tuple[int, list[str]]: the number of merged samples and the clip ids that failed.
    """
    import lmdb

    counts, failed = [], []
    for shard in shard_dirs:
        env = lmdb.open(shard, readonly=True, lock=False)
        with env.begin() as txn:
            counts.append(int(txn.get(b"__count__").decode()))
            failed += [c for c in txn.get(b"__failed__").decode().split("\n") if c]
        env.close()
    total = sum(counts)
    os.makedirs(out_dir, exist_ok=True)
    env = lmdb.open(out_dir, map_size=int(total * PER_SAMPLE_BYTES * 1.3) + 100_000_000)
    keys = list(ARRAY_KEYS) + list(STRING_KEYS)
    merged = 0
    for shard, count in zip(shard_dirs, counts):
        src = lmdb.open(shard, readonly=True, lock=False)
        with src.begin() as rtxn, env.begin(write=True) as wtxn:
            for j in range(count):
                for key in keys:
                    wtxn.put(f"{key}_{merged}_data".encode(), rtxn.get(f"{key}_{j}_data".encode()))
                merged += 1
        src.close()
    with env.begin(write=True) as txn:
        for key, (_, shape) in ARRAY_KEYS.items():
            txn.put(f"{key}_shape".encode(), " ".join(map(str, (total, *shape))).encode())
        for key in STRING_KEYS:
            txn.put(f"{key}_shape".encode(), str(total).encode())
    env.sync()
    env.close()
    return total, failed


def make_loader(dataset: ClipDataset, num_workers: int, pin_memory: bool) -> Any:
    """One item per clip, decoded by ``num_workers`` worker processes, in dataset order.

    Args:
        dataset (ClipDataset): the clips of this rank.
        num_workers (int): decode workers; 0 decodes in the main process.
        pin_memory (bool): pin the frames for an asynchronous copy to the GPU.

    Returns:
        Any: a ``torch.utils.data.DataLoader``.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clip_ids", required=True, help="kept clip ids from spatialvid.py --output")
    p.add_argument("--metadata", required=True, help="SpatialVID_HQ_metadata.csv")
    p.add_argument("--annotations_root", required=True)
    p.add_argument("--videos_root", required=True)
    p.add_argument("--caption_field", default=DEFAULT_CAPTION_FIELD, choices=CAPTION_FIELDS)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vae_path", default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--num_workers", type=int, default=8, help="decode workers per rank")
    return p.parse_args()


def main() -> None:
    """Encode this rank's share of the clips, then merge all shards on rank 0."""
    import lmdb

    from minwm.modeling.wan21.vae import Wan21VAE

    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = 0
    if world_size > 1:
        import datetime

        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=4))
        rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    # fp32 VAE; TF32 convolutions are cuDNN's default, set here so a global change can't slow it.
    torch.backends.cudnn.allow_tf32 = True

    with open(args.clip_ids) as f:
        clip_ids = [line.strip() for line in f if line.strip()]
    rows = read_metadata(args.metadata, clip_ids=set(clip_ids))
    missing = [c for c in clip_ids if c not in rows]
    if missing:
        raise ValueError(f"{len(missing)} clip ids not in the metadata, e.g. {missing[:3]}")
    shard = clip_ids[rank::world_size]
    print(f"rank {rank}: {len(shard)} of {len(clip_ids)} clips", flush=True)

    dataset = ClipDataset(
        [rows[c] for c in shard], args.annotations_root, args.videos_root, args.caption_field
    )
    loader = make_loader(dataset, args.num_workers, pin_memory=True)
    vae = Wan21VAE(vae_pth=args.vae_path, dtype=torch.float32, device=device)
    rank_dir = os.path.join(args.output_dir, f".rank_{rank}")
    os.makedirs(rank_dir, exist_ok=True)
    env = lmdb.open(rank_dir, map_size=int(len(shard) * PER_SAMPLE_BYTES * 1.3) + 100_000_000)
    t0 = time.time()
    count, failed = write_shard(loader, env, vae, device, log_prefix=f"rank {rank}: ")
    env.sync()
    env.close()
    print(
        f"rank {rank}: {count} written, {len(failed)} failed, {time.time() - t0:.0f}s", flush=True
    )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        shard_dirs = [os.path.join(args.output_dir, f".rank_{r}") for r in range(world_size)]
        total, failed = merge_shards(shard_dirs, os.path.join(args.output_dir, "data"))
        with open(os.path.join(args.output_dir, "failed_clip_ids.txt"), "w") as f:
            f.writelines(f"{clip_id}\n" for clip_id in failed)
        for shard_dir in shard_dirs:
            shutil.rmtree(shard_dir)
        print(
            f"merged {total} samples into {os.path.join(args.output_dir, 'data')}; "
            f"{len(failed)} failed (failed_clip_ids.txt)",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
