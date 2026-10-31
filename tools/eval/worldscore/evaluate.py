"""Score packed WorldScore instances with WorldScore's own per-instance metrics.

Runs in the WorldScore environment, one process per GPU: ``--shard i --num-shards n``
takes every n-th unscored instance. Each instance gets WorldScore's
``evaluation.json``; instances that already have one are skipped, so a rerun
resumes. An instance that raises is reported and skipped (exit code 1), not
fatal to the shard. ``summarize.py`` aggregates afterwards::

    CUDA_VISIBLE_DEVICES=0 python tools/eval/worldscore/evaluate.py \\
        --worldscore-root $WS_ROOT --dataset-root $WS_DATA/WorldScore-Dataset \\
        --runs-root $OUT_DIR --shard 0 --num-shards 4
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

from _common import instance_dirs, register_model, worldscore_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worldscore-root", required=True, help="WorldScore checkout")
    parser.add_argument("--dataset-root", required=True, help="the WorldScore-Dataset directory")
    parser.add_argument("--runs-root", required=True, help="this run's output root")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = os.path.abspath(args.runs_root)
    dataset_root = os.path.abspath(args.dataset_root)
    worldscore_root = os.path.abspath(args.worldscore_root)
    register_model()
    # WorldScore loads metric checkpoints from paths relative to its checkout.
    os.chdir(worldscore_root)
    from worldscore.benchmark.helpers.evaluator import Evaluator, process_batch

    failed = []
    for split in ("static", "dynamic"):
        cfg = worldscore_config(worldscore_root, dataset_root, runs_root, split)
        evaluator = Evaluator(cfg)
        aspects = evaluator.build_full_aspect_list()
        todo = [
            d
            for d in instance_dirs(runs_root, split)
            if not os.path.exists(os.path.join(d, "evaluation.json")) and evaluator.data_exists(d)
        ][args.shard :: args.num_shards]
        print(f"[shard {args.shard}] {split}: {len(todo)} instance(s) to score", flush=True)
        for d in todo:
            # WorldScore's path components, e.g. static: [style, scene_type, category, name, dir].
            parts = Path(d).parts[-(4 if split == "static" else 3) :]
            try:
                process_batch(cfg, [[*parts, Path(d)]], aspects, split)
            except Exception:
                traceback.print_exc()
                failed.append(d)

    if failed:
        print(f"[shard {args.shard}] {len(failed)} instance(s) failed:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
