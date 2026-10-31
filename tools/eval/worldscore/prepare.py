"""Materialize a WorldScore subset and write the matching minWM benchmark JSON.

The subset is stratified: from the static split, single-move instances of one
visual style, ``--static-per-group`` per (camera move, indoor/outdoor) for each of
``--moves``; from the dynamic split, ``--dynamic-per-group`` per motion type (0
skips the split). Subsets are nested: for the same seed, a smaller subset is a
subset of a larger one, so a run can start small and grow under the same output
root without regenerating anything. The order is rank-major (every group's first
pick, then every group's second, ...), so a run stopped part way still covers
the groups evenly.

Instance directories are written by WorldScore's own dataloader and adapter
(``camera_data.json``, ``image_data.json``, ``input_image.png``), so prompts and
ground-truth cameras are exactly what its evaluator expects. Each benchmark item
carries WorldScore's prompt, the ground-truth camera converted to minWM poses,
and a seed fixed by the instance id. Instances whose frames are already in place
are skipped, so the benchmark lists only what is left to generate. WorldScore's
code is imported from ``--worldscore-root`` and none of its metrics are, so any
environment with torch, torchvision, scipy and omegaconf runs this, e.g. minWM's::

    python tools/eval/worldscore/prepare.py --worldscore-root $WS_ROOT \\
        --dataset-root $WS_DATA/WorldScore-Dataset --runs-root $OUT_DIR \\
        --bench $OUT_DIR/bench.json
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from _common import (
    CAMERA_MOVES,
    NUM_LATENT_FRAMES,
    OUTPUT_DIR,
    check_motion,
    instance_id,
    instance_seed,
    register_model,
    worldscore_config,
    worldscore_to_minwm_poses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worldscore-root", required=True, help="WorldScore checkout")
    parser.add_argument("--dataset-root", required=True, help="the WorldScore-Dataset directory")
    parser.add_argument("--runs-root", required=True, help="this run's output root")
    parser.add_argument("--bench", required=True, help="where to write the minWM benchmark JSON")
    parser.add_argument("--style", default="photorealistic", choices=["photorealistic", "stylized"])
    parser.add_argument(
        "--moves",
        default=",".join(CAMERA_MOVES),
        help="comma-separated static camera moves to include (default: all 8)",
    )
    parser.add_argument("--static-per-group", type=int, default=6)
    parser.add_argument("--dynamic-per-group", type=int, default=5, help="0 skips the split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.52,
        help="WorldScore's push_in/pull_out travels 1 unit; 1.52 turns that into 19 "
        "latent-frame steps of 0.08, the step minWM was trained on",
    )
    args = parser.parse_args()
    args.moves = [m.strip() for m in args.moves.split(",") if m.strip()]
    unknown = sorted(set(args.moves) - set(CAMERA_MOVES))
    if unknown:
        parser.error(f"unknown move(s) {unknown}; choose from {list(CAMERA_MOVES)}")
    return args


def select_subset(
    items: list[dict], key: Callable[[dict], tuple], per_group: int, seed: int
) -> list[dict]:
    """Pick ``per_group`` items from each group, nested across ``per_group``.

    Each group is shuffled by its own seeded RNG, independent of ``per_group`` and
    of the other groups, and its first ``per_group`` members are taken. So for the
    same seed, asking for more per group or for more groups only adds items.

    Args:
        items (list[dict]): WorldScore instances.
        key (Callable[[dict], tuple]): maps an instance to its group (a tuple of str).
        per_group (int): how many to take from each group (all if fewer).
        seed (int): sampling seed.

    Returns:
        list[dict]: the chosen instances, rank-major: every group's first pick in
        sorted group order, then every group's second, and so on.
    """
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    ranked = []
    for group in sorted(groups):
        members = sorted(groups[group], key=lambda x: x["image"])
        random.Random(f"{seed}/{'/'.join(group)}").shuffle(members)
        ranked += [(rank, group, item) for rank, item in enumerate(members[:per_group])]
    return [item for _, _, item in sorted(ranked, key=lambda r: (r[0], r[1]))]


def write_subset(path: str, items: list[dict]) -> None:
    """Record a split's subset; it may grow on a rerun but not lose instances.

    Raises:
        SystemExit: if ``path`` holds an instance the new subset lacks, i.e. the
            subset shrank or changed (other ``--style`` / ``--seed``).
    """
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if any(item not in items for item in old):
            raise SystemExit(
                f"{path} holds instances the new subset lacks (smaller subset, or changed "
                "--style/--seed?); a subset may only grow under one run, so use a new run name"
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1, ensure_ascii=False)


def materialize(cfg: dict, translation_scale: float) -> list[dict]:
    """Write the not-yet-generated instances of one split; return their benchmark items.

    Args:
        cfg (dict): a :func:`worldscore_config` with ``json_path`` set to the subset.
        translation_scale (float): see :func:`worldscore_to_minwm_poses`.

    Returns:
        list[dict]: ``{id, caption, poses, seed}`` items for minWM's benchmark JSON.
    """
    from worldscore.benchmark.helpers.dataloaders import dataloader_videogen
    from worldscore.benchmark.helpers.helper import Helper

    loader = dataloader_videogen(cfg)
    helper = Helper(cfg)
    out_root = os.path.join(cfg["runs_root"], OUTPUT_DIR)
    items = []
    for data in loader.data:
        (prompt,) = helper.adapt(data)
        out_dir = data["output_dir"]
        (layout,) = data["camera_path"]
        if layout == "fixed":
            poses = np.tile(np.eye(4, dtype=np.float32), (NUM_LATENT_FRAMES, 1, 1))
        else:
            with open(os.path.join(out_dir, "camera_data.json"), encoding="utf-8") as f:
                cameras = json.load(f)["cameras_interp"]
            poses = worldscore_to_minwm_poses(cameras, translation_scale)
            check_motion(layout, poses, translation_scale)
        sample_id = instance_id(os.path.relpath(out_dir, out_root))
        items.append(
            {
                "id": sample_id,
                "caption": prompt,
                "poses": poses.round(6).tolist(),
                "seed": instance_seed(sample_id),
            }
        )
    return items


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.worldscore_root))
    register_model()
    runs_root = os.path.abspath(args.runs_root)
    splits = {
        "static": (
            lambda x: x["visual_style"] == args.style
            and len(x["camera_path"]) == 1
            and x["camera_path"][0] in args.moves,
            lambda x: (x["camera_path"][0], x["scene_type"]),
            args.static_per_group,
        ),
        "dynamic": (
            lambda x: x["visual_style"] == args.style,
            lambda x: (x["motion_type"],),
            args.dynamic_per_group,
        ),
    }
    bench = []
    for split, (keep, group, per_group) in splits.items():
        with open(os.path.join(args.dataset_root, split, f"{split}.json"), encoding="utf-8") as f:
            items = [x for x in json.load(f) if keep(x)]
        subset = select_subset(items, group, per_group, args.seed)
        subset_path = os.path.join(runs_root, "subset", f"{split}.json")
        write_subset(subset_path, subset)
        if not subset:
            continue
        cfg = worldscore_config(args.worldscore_root, args.dataset_root, runs_root, split)
        cfg["json_path"] = subset_path
        pending = materialize(cfg, args.translation_scale)
        print(f"{split}: {len(subset)} in subset, {len(pending)} left to generate", flush=True)
        bench += pending

    with open(args.bench, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
