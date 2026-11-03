"""Write the dense disparity LMDB that pairs with a SpatialVID main LMDB.

Follows ``docs/ai-generated/decisions.md``: section "2026-11-01：① 和 ② 都在 DeltaAI 训" item 3
(disparity lives in its own LMDB, in the main LMDB's sample order) and the **disparity** bullets
of "2026-11-01：SpatialVID 数据格式". Per labelled clip it writes:

    disparity  (20, 1, 240, 416) float16  d in [0, 1], 1 nearest, γ = 1
    clip_ids   str                        SpatialVID clip id, to check against the main LMDB

The layout matches ``build_spatialvid_lmdb.py``: ``<key>_shape`` plus ``<key>_<i>_data``. The
sample order is taken from the main LMDB's ``clip_ids``, not from a clip-id list, because the
main pass shards clips across ranks. Each clip is re-planned with
:func:`~tools.data.wan21.spatialvid.plan_lmdb_clip`, the function the main pass plans with, and
its poses and intrinsics must match the main LMDB's.

Clips whose depth breaks a contract check (:class:`DisparitySkip`) and clips that fail are left
out; the written samples keep the main LMDB's relative order. ``<output_dir>`` gets the LMDB in
``data``, ``skipped_clip_ids.txt`` (id and reason) and ``failed_clip_ids.txt``. The pass stops
with an error once more than 5% of a group's clips are skipped as inverted depth.

Worker processes plan clips and compute labels; the main process writes them in order. CPU only.

Usage (from the repo root, as a module so ``tools.data.wan21`` resolves)::

    python -m tools.data.wan21.build_spatialvid_disparity_lmdb \\
        --main_lmdb /path/spatialvid/train/data \\
        --metadata /path/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv \\
        --annotations_root /path/SpatialVID-HQ/extracted/annotations/SpatialVID \\
        --videos_root /path/SpatialVID-HQ/extracted/videos/SpatialVID \\
        --depths_root /path/SpatialVID-HQ/extracted/depths/SpatialVID \\
        --output_dir /path/spatialvid/train_disparity \\
        --num_workers 32
"""

import argparse
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any

import numpy as np

from tools.data.wan21.spatialvid import NUM_LATENT_FRAMES, plan_lmdb_clip, read_metadata
from tools.data.wan21.spatialvid_disparity import (
    DISPARITY_SHAPE,
    SKIP_INVERTED_DEPTH,
    DisparitySkip,
    clip_disparity,
)

MAX_INVERTED_FRACTION = 0.05
# Poses are re-derived through scipy on another machine; a different window moves them by far more.
POSE_ATOL = 1e-4
PER_SAMPLE_BYTES = int(np.prod(DISPARITY_SHAPE)) * 2 + 1000


def depth_zip_path(row: Mapping, depths_root: str) -> str:
    """``<depths_root>/depths/group_<NNNN>/<clip_id>.zip``, the clip's depth zip.

    Args:
        row (Mapping): the clip's metadata row; uses ``id`` and ``group id``.
        depths_root (str): the extracted SpatialVID depth directory.

    Returns:
        str: path of the zip.
    """
    group = f"group_{int(row['group id']):04d}"
    return os.path.join(depths_root, "depths", group, f"{row['id']}.zip")


def read_main_lmdb(path: str) -> list[dict[str, Any]]:
    """Clip id, poses and intrinsics of every sample of a main LMDB, in sample order.

    Args:
        path (str): the main LMDB directory (``<output_dir>/data`` of the main pass).

    Returns:
        list[dict[str, Any]]: per sample ``clip_ids`` (str), ``poses`` ``(20, 7)`` float32 and
        ``intrinsics`` ``(4,)`` float32.
    """
    import lmdb

    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        count = int(txn.get(b"clip_ids_shape").decode())
        samples = [
            {
                "clip_ids": txn.get(f"clip_ids_{i}_data".encode()).decode(),
                "poses": np.frombuffer(txn.get(f"poses_{i}_data".encode()), np.float32).reshape(
                    NUM_LATENT_FRAMES, 7
                ),
                "intrinsics": np.frombuffer(txn.get(f"intrinsics_{i}_data".encode()), np.float32),
            }
            for i in range(count)
        ]
    env.close()
    return samples


def label_clip(
    job: tuple[Mapping, Mapping], annotations_root: str, videos_root: str, depths_root: str
) -> dict[str, Any]:
    """Plan one main-LMDB sample's clip again and compute its disparity label.

    Never raises: skips and failures come back as the ``status``.

    Args:
        job (tuple[Mapping, Mapping]): the sample from :func:`read_main_lmdb` and the clip's
            metadata row.
        annotations_root (str): directory the ``annotation path`` column is relative to.
        videos_root (str): directory the ``video path`` column is relative to.
        depths_root (str): see :func:`depth_zip_path`.

    Returns:
        dict[str, Any]: ``clip_ids``, ``seconds`` and ``status``: ``"ok"`` with ``disparity``
        and ``stats``, ``"skip"`` with ``reason`` and ``message``, or ``"error"`` with
        ``message``.
    """
    sample, row = job
    clip_id = sample["clip_ids"]
    t0 = time.perf_counter()
    try:
        plan = plan_lmdb_clip(row, annotations_root, videos_root)
        if not (
            np.allclose(plan.poses, sample["poses"], rtol=0, atol=POSE_ATOL)
            and np.allclose(plan.intrinsics, sample["intrinsics"], rtol=0, atol=1e-6)
        ):
            raise ValueError(f"{clip_id}: plan differs from the main LMDB's poses or intrinsics")
        d, stats = clip_disparity(depth_zip_path(row, depths_root), plan)
        out = {"status": "ok", "disparity": d, "stats": stats}
    except DisparitySkip as e:
        out = {"status": "skip", "reason": e.reason, "message": str(e)}
    except Exception as e:
        out = {"status": "error", "message": repr(e)}
    return {**out, "clip_ids": clip_id, "seconds": time.perf_counter() - t0}


class DisparityTally:
    """What a disparity pass wrote, skipped and failed, plus the numbers of its summary.

    Args:
        groups (Mapping[str, str]): group id of every clip of the main LMDB.
    """

    def __init__(self, groups: Mapping[str, str]):
        self.groups = dict(groups)
        self.group_sizes = Counter(self.groups.values())
        self.written = 0
        self.skipped: list[tuple[str, str]] = []
        self.failed: list[str] = []
        self.inverted: Counter = Counter()
        self.clipped_fractions: list[float] = []
        self._sum = self._sum_sq = 0.0
        self._count = 0

    def add_label(self, d: np.ndarray, clipped_fraction: float) -> None:
        """Count one written label: its clipped fraction and the moments of ``2d − 1``."""
        x = 2 * d.astype(np.float32) - 1
        self._sum += float(np.sum(x, dtype=np.float64))
        self._sum_sq += float(np.sum(np.square(x), dtype=np.float64))
        self._count += x.size
        self.clipped_fractions.append(clipped_fraction)
        self.written += 1

    def add_skip(self, clip_id: str, reason: str) -> None:
        """Count one skipped clip.

        Args:
            clip_id (str): the clip.
            reason (str): its ``SKIP_*`` reason.

        Raises:
            RuntimeError: if more than ``MAX_INVERTED_FRACTION`` of the clip's group is now
                skipped as inverted depth, which more likely means the depth format differs
                from the contract than that the scenes are unusual.
        """
        self.skipped.append((clip_id, reason))
        if reason != SKIP_INVERTED_DEPTH:
            return
        group = self.groups[clip_id]
        self.inverted[group] += 1
        if self.inverted[group] > MAX_INVERTED_FRACTION * self.group_sizes[group]:
            raise RuntimeError(
                f"group {group}: {self.inverted[group]} of {self.group_sizes[group]} clips "
                f"skipped as {SKIP_INVERTED_DEPTH} (limit {MAX_INVERTED_FRACTION:.0%}); "
                "check the depth format"
            )

    def std(self) -> float:
        """Standard deviation of ``2d − 1`` over every value of every written label."""
        mean = self._sum / self._count
        return float(np.sqrt(max(self._sum_sq / self._count - mean**2, 0.0)))

    def summary(self) -> list[str]:
        """The pass's closing log lines."""
        reasons = Counter(reason for _, reason in self.skipped)
        by_reason = ", ".join(f"{r}={n}" for r, n in sorted(reasons.items()))
        lines = [
            f"written {self.written}, skipped {len(self.skipped)} ({by_reason or 'none'}), "
            f"failed {len(self.failed)} of {len(self.groups)}"
        ]
        if not self.written:
            return lines
        fractions = np.asarray(self.clipped_fractions)
        std = self.std()
        lines += [
            f"clipped_fraction median={np.median(fractions):.4f} "
            f"P90={np.quantile(fractions, 0.9, method='higher'):.4f}",
            f"std(2d-1)={std:.4f} 1/std={1 / std:.4f} a (1/std to 0.5)={round(2 / std) / 2}",
        ]
        return lines

    def write_lists(self, output_dir: str) -> None:
        """Write ``skipped_clip_ids.txt`` (id and reason) and ``failed_clip_ids.txt``."""
        with open(os.path.join(output_dir, "skipped_clip_ids.txt"), "w") as f:
            f.writelines(f"{clip_id} {reason}\n" for clip_id, reason in self.skipped)
        with open(os.path.join(output_dir, "failed_clip_ids.txt"), "w") as f:
            f.writelines(f"{clip_id}\n" for clip_id in self.failed)


def write_disparity_lmdb(results: Iterable[Mapping], env: Any, tally: DisparityTally) -> None:
    """Write the labelled clips of ``results`` to ``env`` in order, logging every clip.

    The ``<key>_shape`` entries are written last, so an LMDB of a stopped pass has none and
    cannot be read as complete.

    Args:
        results (Iterable[Mapping]): :func:`label_clip` results in main-LMDB order.
        env (Any): the ``lmdb`` environment of the disparity LMDB.
        tally (DisparityTally): updated with every result.

    Raises:
        RuntimeError: from :meth:`DisparityTally.add_skip`.
    """
    total = len(tally.groups)
    for n, result in enumerate(results, 1):
        clip_id = result["clip_ids"]
        head = f"{n}/{total} {clip_id}"
        if result["status"] == "skip":
            print(f"{head} SKIP {result['message']} {result['seconds']:.2f}s", flush=True)
            tally.add_skip(clip_id, result["reason"])
            continue
        if result["status"] == "error":
            print(f"{head} failed: {result['message']}", flush=True)
            tally.failed.append(clip_id)
            continue
        d, stats = result["disparity"], result["stats"]
        if d.shape != DISPARITY_SHAPE or d.dtype != np.float16:
            print(f"{head} failed: disparity is {d.dtype}{d.shape}", flush=True)
            tally.failed.append(clip_id)
            continue
        with env.begin(write=True) as txn:
            txn.put(f"disparity_{tally.written}_data".encode(), np.ascontiguousarray(d).tobytes())
            txn.put(f"clip_ids_{tally.written}_data".encode(), clip_id.encode())
        tally.add_label(d, stats.clipped_fraction)
        print(
            f"{head} p90={stats.p90:.4f} clipped={stats.clipped_fraction:.4f} "
            f"near_below={stats.near_below_fraction:.2f} {result['seconds']:.2f}s",
            flush=True,
        )
    with env.begin(write=True) as txn:
        shape = " ".join(map(str, (tally.written, *DISPARITY_SHAPE)))
        txn.put(b"disparity_shape", shape.encode())
        txn.put(b"clip_ids_shape", str(tally.written).encode())


def main(argv: list[str] | None = None) -> None:
    """Label every clip of the main LMDB and write the disparity LMDB.

    Args:
        argv (list[str] | None): command-line arguments; ``None`` reads ``sys.argv``.

    Raises:
        ValueError: if a main-LMDB clip id is not in the metadata.
        RuntimeError: from the inverted-depth stop rule.
    """
    import lmdb

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--main_lmdb", required=True, help="main LMDB directory, e.g. .../train/data")
    p.add_argument("--metadata", required=True, help="SpatialVID_HQ_metadata.csv")
    p.add_argument("--annotations_root", required=True)
    p.add_argument("--videos_root", required=True)
    p.add_argument("--depths_root", required=True, help="directory holding depths/group_NNNN/")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_workers", type=int, default=32, help="0 labels in the main process")
    args = p.parse_args(argv)

    samples = read_main_lmdb(args.main_lmdb)
    rows = read_metadata(args.metadata, clip_ids={s["clip_ids"] for s in samples})
    missing = [s["clip_ids"] for s in samples if s["clip_ids"] not in rows]
    if missing:
        raise ValueError(f"{len(missing)} clip ids not in the metadata, e.g. {missing[:3]}")
    tally = DisparityTally({s["clip_ids"]: rows[s["clip_ids"]]["group id"] for s in samples})
    print(f"{len(samples)} clips in {args.main_lmdb}, groups {dict(tally.group_sizes)}")

    worker = partial(
        label_clip,
        annotations_root=args.annotations_root,
        videos_root=args.videos_root,
        depths_root=args.depths_root,
    )
    jobs = ((s, rows[s["clip_ids"]]) for s in samples)
    os.makedirs(args.output_dir, exist_ok=True)
    env = lmdb.open(
        os.path.join(args.output_dir, "data"),
        map_size=int(len(samples) * PER_SAMPLE_BYTES * 1.3) + 100_000_000,
    )
    t0 = time.time()
    try:
        if args.num_workers > 0:
            from multiprocessing import Pool

            with Pool(args.num_workers) as pool:
                write_disparity_lmdb(pool.imap(worker, jobs), env, tally)
        else:
            write_disparity_lmdb(map(worker, jobs), env, tally)
    finally:
        env.sync()
        env.close()
        tally.write_lists(args.output_dir)
    for line in tally.summary():
        print(line)
    print(f"{time.time() - t0:.0f}s; disparity LMDB in {os.path.join(args.output_dir, 'data')}")


if __name__ == "__main__":
    main()
