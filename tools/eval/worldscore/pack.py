"""Move minWM outputs into their WorldScore instance directories.

For each ``{id}.frames/`` that ``tools/infer_mwm.py`` wrote (``inference.save_frames=True``),
moves the frames to ``<instance>/frames/`` and the mp4 to ``<instance>/videos/output.mp4``,
the layout WorldScore's evaluator reads. Safe to rerun: outputs already moved are gone
from the generation directory, so it also picks up what a preempted job left behind::

    python tools/eval/worldscore/pack.py --gen-dir $OUT_DIR/gen --runs-root $OUT_DIR
"""

import argparse
import glob
import json
import os
import shutil

from _common import OUTPUT_DIR, instance_rel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gen-dir", required=True, help="minWM inference.output_dir")
    parser.add_argument("--runs-root", required=True, help="this run's output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packed = 0
    for frames_dir in sorted(glob.glob(os.path.join(args.gen_dir, "*.frames"))):
        sample_id = os.path.basename(frames_dir)[: -len(".frames")]
        instance = os.path.join(args.runs_root, OUTPUT_DIR, instance_rel_path(sample_id))
        with open(os.path.join(instance, "image_data.json"), encoding="utf-8") as f:
            expected = json.load(f)["total_frames"]
        num_frames = len(glob.glob(os.path.join(frames_dir, "*.png")))
        if num_frames != expected:
            raise SystemExit(
                f"{frames_dir} has {num_frames} frames but WorldScore expects {expected}; "
                "is the minWM config's num_frames the one tools/eval/worldscore/_common.py assumes?"
            )
        shutil.rmtree(os.path.join(instance, "frames"), ignore_errors=True)
        os.replace(frames_dir, os.path.join(instance, "frames"))
        video = os.path.join(args.gen_dir, f"{sample_id}.mp4")
        if os.path.exists(video):
            os.makedirs(os.path.join(instance, "videos"), exist_ok=True)
            os.replace(video, os.path.join(instance, "videos", "output.mp4"))
        packed += 1
    print(f"packed {packed} instance(s) from {args.gen_dir}", flush=True)


if __name__ == "__main__":
    main()
