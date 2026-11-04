"""Check a disparity model's own d against a monocular estimate from its generated RGB.

For every packed instance with a ``disparity.pt`` (baseline 2 and later; other runs have
none and are skipped), runs Depth Anything V2 Small on the generated frames and compares
it with the model's d, frame by frame, in ``<instance>/disparity_consistency.json``.

- **Frames**: d has one map per latent frame; latent frame k is compared with pixel frame
  4k, the frame its pose belongs to (``worldscore_to_minwm_poses``). The estimate is
  area-resized to d's 2-pixel grid.
- **Fit**: Depth Anything V2 outputs affine-invariant inverse depth (its loss is MiDaS's
  scale- and shift-invariant one), so its zero is arbitrary. Each frame's estimate D is
  fitted to d by least squares as ``r = s * D + t`` (``--fit affine``, the default).
  ``--fit scale`` (``t = 0``) is the right fit only against an up-to-scale reference,
  whose zero is at infinity like d's; against this estimator it charges the estimator's
  arbitrary shift to the model.
- **Pixels**: d is clipped at 1, so pixels with ``d >= 0.999`` are left out, as are pixels
  with d or r below :data:`DISPARITY_FLOOR` (sky and far background: d = 0.9 (1/z) / P90,
  so depth beyond 45x the depth at the first chunk's P90 disparity, where depth ratios
  blow up).
- **Metrics**, with r as the reference: AbsRel in depth, ``mean |1/d - 1/r| / (1/r)``,
  and delta1, the share of pixels with ``max(d/r, r/d) < 1.25`` (the same in depth and
  disparity). Each frame also records its fit and the share of pixels kept.

Runs in the minWM environment (transformers); instances already scored with the same
fit are skipped::

    python tools/eval/worldscore/score_disparity.py --runs-root $OUT_DIR
"""

import argparse
import glob
import json
import os
from collections.abc import Callable

import numpy as np
from _common import instance_dirs

ESTIMATOR = "depth-anything/Depth-Anything-V2-Small-hf"
RESULT_FILE = "disparity_consistency.json"
DISPARITY_FLOOR = 0.02
SATURATED = 0.999
DELTA = 1.25
MIN_PIXELS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", required=True, help="this run's output root")
    parser.add_argument("--fit", default="affine", choices=["affine", "scale"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def frame_metrics(d: np.ndarray, estimate: np.ndarray, fit: str) -> dict:
    """Fit one frame's estimate to the model's d and score d against it.

    Args:
        d (np.ndarray): ``(H, W)`` model disparity in ``[0, 1]``.
        estimate (np.ndarray): ``(H, W)`` estimator inverse depth on the same grid.
        fit (str): ``"affine"`` (scale and shift) or ``"scale"``.

    Returns:
        dict: ``abs_rel``, ``delta1`` (``None`` if fewer than :data:`MIN_PIXELS` are
        kept), ``scale``, ``shift`` and ``valid_fraction``.
    """
    valid = (d > DISPARITY_FLOOR) & (d < SATURATED) & np.isfinite(estimate)
    x, y = estimate[valid].astype(np.float64), d[valid].astype(np.float64)
    scale, shift, abs_rel, delta1 = float("nan"), 0.0, None, None
    if len(x) >= MIN_PIXELS:
        if fit == "affine":
            (scale, shift), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
        else:
            scale = float(np.sum(x * y) / np.sum(x * x))
        ref = scale * estimate + shift
        valid &= ref > DISPARITY_FLOOR
    if valid.sum() >= MIN_PIXELS:
        dv, rv = d[valid].astype(np.float64), ref[valid]
        abs_rel = float(np.mean(np.abs(rv - dv) / dv))
        delta1 = float(np.mean(np.maximum(dv / rv, rv / dv) < DELTA))
    return {
        "abs_rel": abs_rel,
        "delta1": delta1,
        "scale": float(scale),
        "shift": float(shift),
        "valid_fraction": float(valid.mean()),
    }


def score_instance(d: np.ndarray, frame_paths: list[str], estimate: Callable, fit: str) -> dict:
    """Score one instance's ``(F, H, W)`` disparity against its pixel frames.

    Args:
        d (np.ndarray): ``(F, H, W)`` model disparity, one map per latent frame.
        frame_paths (list[str]): the instance's sorted pixel-frame PNGs.
        estimate (Callable): maps a list of frame paths and an ``(H, W)`` size to
            ``(N, H, W)`` inverse depth (see :class:`DepthAnything`).
        fit (str): ``"affine"`` or ``"scale"``.

    Returns:
        dict: the ``disparity_consistency.json`` payload.

    Raises:
        ValueError: if the frame count does not match ``4 * (F - 1) + 1``.
    """
    num_latent = len(d)
    if len(frame_paths) != 4 * (num_latent - 1) + 1:
        raise ValueError(f"{len(frame_paths)} frames do not match {num_latent} disparity maps")
    pixel_frames = [4 * k for k in range(num_latent)]
    estimates = estimate([frame_paths[i] for i in pixel_frames], d.shape[1:])
    frames = [
        {"latent_frame": k, "pixel_frame": i, **frame_metrics(d[k], estimates[k], fit)}
        for k, i in enumerate(pixel_frames)
    ]

    def mean(key: str) -> float | None:
        values = [f[key] for f in frames if f[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "estimator": ESTIMATOR,
        "fit": fit,
        "disparity_floor": DISPARITY_FLOOR,
        "abs_rel": mean("abs_rel"),
        "delta1": mean("delta1"),
        "frames": frames,
    }


class DepthAnything:
    """Depth Anything V2 relative inverse depth, area-resized to a target grid.

    Args:
        device (str): torch device.
        model_id (str): Hugging Face checkpoint.
    """

    def __init__(self, device: str, model_id: str = ESTIMATOR) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def __call__(self, frame_paths: list[str], size: tuple[int, int]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            depth = self.model(**inputs).predicted_depth.float()
        return F.interpolate(depth[:, None], size=tuple(size), mode="area")[:, 0].cpu().numpy()


def main() -> None:
    args = parse_args()
    runs_root = os.path.abspath(args.runs_root)
    todo = []
    for split in ("static", "dynamic"):
        for d in instance_dirs(runs_root, split):
            if not os.path.exists(os.path.join(d, "disparity.pt")):
                continue
            result = os.path.join(d, RESULT_FILE)
            if os.path.exists(result):
                with open(result, encoding="utf-8") as f:
                    if json.load(f).get("fit") == args.fit:
                        continue
            todo.append(d)
    print(f"disparity consistency: {len(todo)} instance(s) to score", flush=True)
    if not todo:
        return

    import torch

    estimate = DepthAnything(args.device)
    for d in todo:
        disparity = torch.load(os.path.join(d, "disparity.pt"), map_location="cpu")[:, 0].numpy()
        frames = sorted(glob.glob(os.path.join(d, "frames", "*.png")))
        payload = score_instance(disparity, frames, estimate, args.fit)
        path = os.path.join(d, RESULT_FILE)
        with open(f"{path}.tmp", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        os.replace(f"{path}.tmp", path)
        print(f"{os.path.relpath(d, runs_root)}: AbsRel {payload['abs_rel']}", flush=True)


if __name__ == "__main__":
    main()
