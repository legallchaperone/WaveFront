"""Upload a run's exported checkpoints to a Hugging Face model repo.

Runs after ``tools/auto_dump.py --local``, which writes
``<output-dir>/ckpts/exported/model_{step}.safetensors``. Each requested step is
uploaded to ``<repo>/<run name>/model_{step}.safetensors``, where the run name is
the last component of ``--output-dir``. Files already in the repo are skipped, so
re-running after a preempted job is safe. Every local file is checked before the
first upload, so a missing export fails without a partial upload.

The token is never passed in: ``huggingface_hub`` reads the one saved by
``hf auth login`` (or the ``HF_TOKEN`` env var).

Usage::

    python tools/upload_hf.py --output-dir $RUN --steps 1000 2000 3000 \\
        --repo cwx-umich/wavefront-checkpoints
"""

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.paths import ckpt_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(description="Upload exported checkpoints to Hugging Face.")
    parser.add_argument("--output-dir", required=True, help="training output_dir (run name)")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="steps to upload")
    parser.add_argument("--repo", required=True, help="model repo id, e.g. user/name")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="directory holding model_{step}.safetensors (default <output-dir>/ckpts/exported)",
    )
    return parser.parse_args(argv)


def upload_checkpoints(
    api: Any, repo: str, run_name: str, export_dir: str, steps: list[int]
) -> dict[int, str]:
    """Upload ``model_{step}.safetensors`` for each step, skipping ones already there.

    Args:
        api (Any): a ``huggingface_hub.HfApi`` (or a stand-in with ``file_exists`` /
            ``upload_file``).
        repo (str): target model repo id.
        run_name (str): folder in the repo for this run.
        export_dir (str): local directory with the exported weights.
        steps (list[int]): checkpoint steps to upload.

    Returns:
        dict[int, str]: step -> URL of the file in the repo.

    Raises:
        FileNotFoundError: if any requested export is missing locally.
    """
    local = {step: os.path.join(export_dir, f"model_{step}.safetensors") for step in steps}
    missing = [path for path in local.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"exported weights not found: {missing}")

    urls = {}
    for step, path in local.items():
        path_in_repo = f"{run_name}/model_{step}.safetensors"
        if api.file_exists(repo, path_in_repo, repo_type="model"):
            print(f"[upload_hf] step {step}: already in {repo}, skipping", flush=True)
        else:
            print(f"[upload_hf] step {step}: uploading {path}", flush=True)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=repo,
                repo_type="model",
                commit_message=f"{run_name}: step {step}",
            )
        urls[step] = f"https://huggingface.co/{repo}/blob/main/{path_in_repo}"
        print(f"[upload_hf] step {step}: {urls[step]}", flush=True)
    return urls


def main(argv: list[str] | None = None) -> None:
    """Upload the requested steps of one run (``argv`` defaults to ``sys.argv[1:]``)."""
    from huggingface_hub import HfApi

    args = parse_args(argv)
    run_name = os.path.basename(os.path.normpath(args.output_dir))
    export_dir = args.export_dir or os.path.join(ckpt_dir(args.output_dir), "exported")
    upload_checkpoints(HfApi(), args.repo, run_name, export_dir, args.steps)


if __name__ == "__main__":
    main()
