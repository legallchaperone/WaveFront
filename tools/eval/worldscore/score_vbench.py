"""Score a WorldScore run's generated videos on VBench quality dimensions.

Runs VBench's own evaluator in its ``custom_input`` mode on every packed instance's
``videos/output.mp4`` and writes the per-video scores to ``<instance>/vbench.json``
(``{dimension: score}``, higher is better). Instances that already hold every requested
dimension are skipped, so a rerun resumes. The default dimensions are quality alarms that
VBench supports for custom videos and WorldScore does not already cover:

- ``subject_consistency``: DINO feature similarity of each frame to the first and the
  previous one (identity drift).
- ``background_consistency``: the same with CLIP features (scene drift).
- ``motion_smoothness``: how well AMT-S interpolates the dropped odd frames (jitter).
- ``imaging_quality``: MUSIQ per frame (blur, noise, over-exposure).

``aesthetic_quality`` is left out: it is the LAION predictor WorldScore's
``clip_aesthetic`` already reports. Camera motion lowers both consistencies on its own,
so compare runs on the same instances, not against VBench's leaderboard.

Runs in the VBench environment (``install_vbench.sh``); with ``--prefetch`` it only
downloads the weights, so jobs need no internet::

    VBENCH_CACHE_DIR=$WORK/.cache/vbench $VB_PYTHON tools/eval/worldscore/score_vbench.py \\
        --runs-root $OUT_DIR
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from _common import OUTPUT_DIR, instance_dirs, instance_id

DEFAULT_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "imaging_quality",
)
RESULT_FILE = "vbench.json"
_RUN_NAME = "vbench"
# VBench reports imaging_quality as the mean MUSIQ score / 100 but keeps each video's
# score on MUSIQ's 0-100 scale; rescale so per-video scores match the dimension's.
_PER_VIDEO_SCALE = {"imaging_quality": 0.01}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", help="this run's output root")
    parser.add_argument("--dimensions", default=",".join(DEFAULT_DIMENSIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefetch", action="store_true", help="only download the weights")
    args = parser.parse_args()
    args.dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
    if not args.prefetch and not args.runs_root:
        parser.error("--runs-root is required unless --prefetch")
    return args


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pending_instances(runs_root: str, dimensions: list[str]) -> dict[str, str]:
    """Generated instances that lack any of ``dimensions``.

    Args:
        runs_root (str): the run's output root.
        dimensions (list[str]): VBench dimensions to score.

    Returns:
        dict[str, str]: instance id (see :func:`instance_id`) -> instance directory.
    """
    out_root = os.path.join(runs_root, OUTPUT_DIR)
    pending = {}
    for split in ("static", "dynamic"):
        for d in instance_dirs(runs_root, split):
            if not os.path.exists(os.path.join(d, "videos", "output.mp4")):
                continue
            done = _read_json(os.path.join(d, RESULT_FILE))
            if not all(dim in done for dim in dimensions):
                pending[instance_id(os.path.relpath(d, out_root))] = d
    return pending


def per_video_scores(results: dict) -> dict[str, dict[str, float]]:
    """Regroup VBench's ``*_eval_results.json`` by video.

    Args:
        results (dict): ``{dimension: [mean, [{"video_path", "video_results", ...}]]}``.

    Returns:
        dict[str, dict[str, float]]: video file stem -> ``{dimension: score}``, on the
        scale of VBench's dimension score.
    """
    scores = defaultdict(dict)
    for dimension, (_, per_video) in results.items():
        scale = _PER_VIDEO_SCALE.get(dimension, 1.0)
        for item in per_video:
            score = float(item["video_results"]) * scale
            scores[Path(item["video_path"]).stem][dimension] = score
    return dict(scores)


def run_vbench(
    instances: dict[str, str], dimensions: list[str], device: str, work_dir: str
) -> dict[str, dict[str, float]]:
    """Score instances' videos with VBench's ``custom_input`` mode.

    VBench names videos after their files, so each instance's ``output.mp4`` is linked
    into one directory as ``<instance id>.mp4``.

    Args:
        instances (dict[str, str]): instance id -> instance directory.
        dimensions (list[str]): VBench dimensions to score.
        device (str): torch device.
        work_dir (str): scratch directory for the links and VBench's own outputs.

    Returns:
        dict[str, dict[str, float]]: instance id -> ``{dimension: score}``.
    """
    import vbench

    videos = os.path.join(work_dir, "videos")
    os.makedirs(videos)
    for sample_id, d in instances.items():
        target = os.path.abspath(os.path.join(d, "videos", "output.mp4"))
        os.symlink(target, os.path.join(videos, f"{sample_id}.mp4"))
    # full_info_dir is read only in VBench's standard-prompt mode.
    full_info = os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")
    evaluator = vbench.VBench(device, full_info, work_dir)
    evaluator.evaluate(
        videos_path=videos,
        name=_RUN_NAME,
        dimension_list=dimensions,
        local=True,
        mode="custom_input",
    )
    return per_video_scores(_read_json(os.path.join(work_dir, f"{_RUN_NAME}_eval_results.json")))


def write_scores(instances: dict[str, str], scores: dict[str, dict[str, float]]) -> None:
    """Merge each instance's new scores into its ``vbench.json``.

    Raises:
        SystemExit: if VBench returned no score for an instance it was given.
    """
    missing = sorted(set(instances) - set(scores))
    if missing:
        raise SystemExit(f"VBench returned no scores for: {missing}")
    for sample_id, d in instances.items():
        path = os.path.join(d, RESULT_FILE)
        merged = {**_read_json(path), **scores[sample_id]}
        with open(f"{path}.tmp", "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1)
        os.replace(f"{path}.tmp", path)


def prefetch(dimensions: list[str]) -> None:
    """Download every weight the dimensions load, into VBench's cache and torch hub."""
    import torch
    from vbench.utils import init_submodules

    submodules = init_submodules(dimensions, local=True)
    if "subject_consistency" in submodules:
        # DINO's hubconf fetches its weights only when the model is built.
        torch.hub.load(**submodules["subject_consistency"])


def main() -> None:
    args = parse_args()
    if args.prefetch:
        prefetch(args.dimensions)
        return
    runs_root = os.path.abspath(args.runs_root)
    instances = pending_instances(runs_root, args.dimensions)
    print(f"vbench: {len(instances)} instance(s) to score", flush=True)
    if not instances:
        return
    with tempfile.TemporaryDirectory(dir=runs_root, prefix="vbench-") as work_dir:
        scores = run_vbench(instances, args.dimensions, args.device, work_dir)
    write_scores(instances, scores)


if __name__ == "__main__":
    main()
