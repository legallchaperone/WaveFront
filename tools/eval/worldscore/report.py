"""Write a static HTML report that puts several runs side by side, case by case.

For runs evaluated on the same WorldScore instances (e.g. the original minWM, baseline 1
and baseline 2, each one ``run.sh`` output root), writes ``<out>/index.html`` with:

- per camera move (and the parallax / no-parallax / all groups ``summarize.py`` uses),
  each run's mean of every per-case metric and, for every run but the first (the
  baseline), the **paired** mean difference to the baseline over the cases both have,
  with a 95% bootstrap CI over cases;
- per case, the prompt, camera move and seed, every run's video (and, for disparity
  models, the colour-mapped d next to it, near = red) with a button that plays the row
  in sync, and the case's metrics: WorldScore's raw camera rotation / translation error,
  reprojection error (3D consistency) and flow AEPE, the VBench dimensions of
  ``score_vbench.py`` and the disparity consistency of ``score_disparity.py``.

Every instance's seed is fixed by its id (:func:`instance_seed`), so all runs start each
case from the same noise; a run whose ``gen/manifest.json`` records another seed is
flagged. Videos are re-encoded from the lossless frames as H.264 (yuv420p, plays in any
browser) under ``<out>/media/``, skipping those already there, so the folder can be
copied and opened anywhere. ``report.json`` holds the same numbers. Runs in the minWM
environment (imageio-ffmpeg, OpenCV, torch for ``disparity.pt``)::

    python tools/eval/worldscore/report.py --out ./local/report \\
        --run minwm=$OUTS/ws-minwm --run b1=$OUTS/ws-b1 --run b2=$OUTS/ws-b2
"""

import argparse
import glob
import html
import json
import os

import numpy as np
from _common import FPS, OUTPUT_DIR, instance_dirs, instance_id, instance_seed
from score_vbench import DEFAULT_DIMENSIONS as VBENCH_DIMENSIONS
from summarize import CAMERA_MOVES, DYNAMIC_RAW, NO_PARALLAX, STATIC_RAW, raw_means

# name -> True if lower is better, False if higher is better, None if neither.
METRICS = {
    **{name: True for name in STATIC_RAW},
    "motion acc": False,
    "flow mag": None,
    **{f"vbench {name}": False for name in VBENCH_DIMENSIONS},
    "disp AbsRel": True,
    "disp δ<1.25": False,
}
PARALLAX_GROUP = "有视差（push/pull/move/orbit）"
NO_PARALLAX_GROUP = "无视差（pan）"
ALL_GROUP = "全部"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=RUNS_ROOT",
        help="a run to show, in display order; the first is the baseline of the differences",
    )
    parser.add_argument("--out", required=True, help="report directory")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap resamples")
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        name, sep, root = spec.partition("=")
        if not sep or not name or name in runs:
            parser.error(f"--run {spec!r}: expected a new NAME=RUNS_ROOT")
        runs[name] = os.path.abspath(root)
    args.runs = runs
    return args


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_metrics(instance: str, split: str) -> dict[str, float | None]:
    """Every per-case metric of one instance (``None`` where not scored).

    Args:
        instance (str): the instance directory.
        split (str): ``"static"`` or ``"dynamic"``.

    Returns:
        dict[str, float | None]: one value per :data:`METRICS` name; WorldScore's error
        sentinels count as not scored, as in ``summarize.py``.
    """
    metrics = dict.fromkeys(METRICS)
    evaluation = _read_json(os.path.join(instance, "evaluation.json"))
    if evaluation is not None:
        means, _ = raw_means([evaluation], STATIC_RAW if split == "static" else DYNAMIC_RAW)
        metrics.update(means)
    vbench = _read_json(os.path.join(instance, "vbench.json")) or {}
    metrics.update({f"vbench {k}": v for k, v in vbench.items() if f"vbench {k}" in METRICS})
    disparity = _read_json(os.path.join(instance, "disparity_consistency.json")) or {}
    metrics["disp AbsRel"] = disparity.get("abs_rel")
    metrics["disp δ<1.25"] = disparity.get("delta1")
    return metrics


def load_run(runs_root: str) -> dict[str, dict]:
    """Collect one run's generated instances.

    Args:
        runs_root (str): the run's output root.

    Returns:
        dict[str, dict]: instance id -> ``{split, group, scene_type, category, prompt,
        dir, frames, disparity, metrics}``; ``frames`` lists the lossless PNGs and
        ``disparity`` is the ``disparity.pt`` path or ``None``.
    """
    out_root = os.path.join(runs_root, OUTPUT_DIR)
    cases = {}
    for split in ("static", "dynamic"):
        for d in instance_dirs(runs_root, split):
            frames = sorted(glob.glob(os.path.join(d, "frames", "*.png")))
            if not frames:
                continue
            image_data = _read_json(os.path.join(d, "image_data.json"))
            prompts = image_data.get("prompt_list") or [""]
            disparity = os.path.join(d, "disparity.pt")
            cases[instance_id(os.path.relpath(d, out_root))] = {
                "split": split,
                "group": (
                    image_data["camera_path"][0] if split == "static" else image_data["motion_type"]
                ),
                "scene_type": image_data.get("scene_type", ""),
                "category": image_data.get("category", ""),
                # A static instance's first prompt is the input image's; the video's is next.
                "prompt": prompts[1] if split == "static" and len(prompts) > 1 else prompts[0],
                "dir": d,
                "frames": frames,
                "disparity": disparity if os.path.exists(disparity) else None,
                "metrics": case_metrics(d, split),
            }
    return cases


def seed_mismatches(runs_root: str) -> list[str]:
    """Ids whose seed in the run's ``gen/manifest.json`` is not :func:`instance_seed`."""
    manifest = _read_json(os.path.join(runs_root, "gen", "manifest.json")) or {}
    return sorted(
        item["id"]
        for item in manifest.get("items", [])
        if item.get("seed") is not None and item["seed"] != instance_seed(str(item["id"]))
    )


def paired_difference(
    baseline: list[float | None], other: list[float | None], resamples: int, seed: int = 0
) -> dict | None:
    """Mean paired difference ``other - baseline`` with a 95% bootstrap CI over cases.

    Args:
        baseline (list[float | None]): the baseline's value per case.
        other (list[float | None]): the other run's value for the same cases.
        resamples (int): bootstrap resamples.
        seed (int): bootstrap RNG seed, fixed so a report is reproducible.

    Returns:
        dict | None: ``{n, mean, lo, hi}`` over the cases both runs scored, or ``None``
        if there are none.
    """
    diffs = np.array(
        [o - b for b, o in zip(baseline, other) if b is not None and o is not None], dtype=float
    )
    if not len(diffs):
        return None
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, len(diffs), size=(resamples, len(diffs)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(diffs), "mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi)}


def case_groups(cases: dict[str, dict]) -> dict[str, list[str]]:
    """Summary groups, in report order: each move, then the parallax groups and all."""
    by_group: dict[str, list[str]] = {}
    for sample_id, case in sorted(cases.items()):
        by_group.setdefault(case["group"], []).append(sample_id)
    moves = [m for m in CAMERA_MOVES if m in by_group]
    groups = {m: by_group[m] for m in moves}
    groups.update({g: by_group[g] for g in sorted(by_group) if g not in groups})
    static = [i for m in moves for i in by_group[m]]
    if static:
        groups[PARALLAX_GROUP] = [i for m in moves if m not in NO_PARALLAX for i in by_group[m]]
        groups[NO_PARALLAX_GROUP] = [i for m in moves if m in NO_PARALLAX for i in by_group[m]]
    groups[ALL_GROUP] = sorted(cases)
    return {name: ids for name, ids in groups.items() if ids}


def summarize_runs(
    runs: dict[str, dict[str, dict]], groups: dict[str, list[str]], resamples: int
) -> dict[str, dict[str, dict]]:
    """Per group and metric: each run's mean and each non-baseline run's paired difference.

    Returns:
        dict[str, dict[str, dict]]: group -> metric -> ``{"mean": {run: (mean, n)},
        "diff": {run: paired_difference(...)}}``; metrics no run scored are left out.
    """
    names = list(runs)
    out = {}
    for group, ids in groups.items():
        out[group] = {}
        for metric in METRICS:
            values = {
                name: [runs[name].get(i, {}).get("metrics", {}).get(metric) for i in ids]
                for name in names
            }
            if all(v is None for vs in values.values() for v in vs):
                continue
            means = {}
            for name, vs in values.items():
                scored = [v for v in vs if v is not None]
                means[name] = (float(np.mean(scored)) if scored else None, len(scored))
            diffs = {
                name: paired_difference(values[names[0]], values[name], resamples)
                for name in names[1:]
            }
            out[group][metric] = {"mean": means, "diff": diffs}
    return out


def latent_frame_per_pixel_frame(num_pixel_frames: int) -> np.ndarray:
    """Latent frame whose VAE group holds each pixel frame: 0 -> 0, 4k-3..4k -> k."""
    return np.ceil(np.arange(num_pixel_frames) / 4).astype(int)


def colorize_disparity(d: np.ndarray) -> np.ndarray:
    """Map ``(N, H, W)`` d in ``[0, 1]`` to ``(N, H, W, 3)`` uint8 RGB (turbo, near = red)."""
    import cv2

    gray = (np.clip(d, 0, 1) * 255).round().astype(np.uint8)
    return np.stack([cv2.applyColorMap(g, cv2.COLORMAP_TURBO)[..., ::-1] for g in gray])


def write_mp4(frames, path: str, fps: int = FPS) -> None:
    """Write RGB uint8 frames (an iterable of ``(H, W, 3)``) as browser-playable H.264."""
    import imageio.v2 as imageio

    tmp = f"{path}.tmp.mp4"
    with imageio.get_writer(
        tmp, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8, macro_block_size=16
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    os.replace(tmp, path)


def encode_case(case: dict, media_dir: str, sample_id: str) -> dict[str, str | None]:
    """Encode a case's RGB (and colour-mapped disparity) videos unless already there.

    Returns:
        dict[str, str | None]: ``video`` and ``disparity`` paths (the latter ``None``
        without ``disparity.pt``).
    """
    os.makedirs(media_dir, exist_ok=True)
    video = os.path.join(media_dir, f"{sample_id}.mp4")
    if not os.path.exists(video):
        from PIL import Image

        write_mp4((np.asarray(Image.open(p).convert("RGB")) for p in case["frames"]), video)
    if case["disparity"] is None:
        return {"video": video, "disparity": None}
    disparity = os.path.join(media_dir, f"{sample_id}.disparity.mp4")
    if not os.path.exists(disparity):
        import torch

        d = torch.load(case["disparity"], map_location="cpu")[:, 0].numpy()
        per_pixel = d[latent_frame_per_pixel_frame(len(case["frames"]))]
        write_mp4(colorize_disparity(per_pixel), disparity)
    return {"video": video, "disparity": disparity}


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.4g}"


def _diff_cell(diff: dict | None, lower_better: bool | None) -> str:
    if diff is None:
        return "<td>–</td>"
    verdict = ""
    if lower_better is not None and diff["n"] >= 2 and (diff["lo"] > 0 or diff["hi"] < 0):
        better = (diff["hi"] < 0) == lower_better
        verdict = " class='better'" if better else " class='worse'"
    return (
        f"<td{verdict}>{diff['mean']:+.3g} <small>[{diff['lo']:+.3g}, {diff['hi']:+.3g}]"
        f" n={diff['n']}</small></td>"
    )


def summary_html(summary: dict, names: list[str]) -> str:
    """Summary tables, one per group."""
    parts = []
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    head += "".join(f"<th>{html.escape(n)} − {html.escape(names[0])}</th>" for n in names[1:])
    for group, metrics in summary.items():
        rows = []
        for metric, stats in metrics.items():
            lower = METRICS[metric]
            arrow = {True: " ↓", False: " ↑", None: ""}[lower]
            cells = "".join(
                f"<td>{_fmt(m)} <small>n={n}</small></td>" for m, n in stats["mean"].values()
            )
            cells += "".join(_diff_cell(stats["diff"][n], lower) for n in names[1:])
            rows.append(f"<tr><th>{html.escape(metric)}{arrow}</th>{cells}</tr>")
        parts.append(
            f"<h3>{html.escape(group)}</h3><table><tr><th>metric</th>{head}</tr>"
            + "".join(rows)
            + "</table>"
        )
    return "\n".join(parts)


def case_html(sample_id: str, runs: dict[str, dict], media: dict[str, dict], out: str) -> str:
    """One case: prompt, videos side by side, and its metrics per run."""
    present = [n for n in runs if sample_id in runs[n]]
    case = runs[present[0]][sample_id]
    anchor = html.escape(sample_id)
    figures = []
    for name in present:
        paths = media[name][sample_id]
        for label, key in ((name, "video"), (f"{name} d", "disparity")):
            if paths[key] is None:
                continue
            src = html.escape(os.path.relpath(paths[key], out))
            figures.append(
                f"<figure><video src='{src}' controls muted loop playsinline preload='metadata'>"
                "</video>"
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
    metric_names = [
        m for m in METRICS if any(runs[n][sample_id]["metrics"][m] is not None for n in present)
    ]
    head = "".join(f"<th>{html.escape(n)}</th>" for n in present)
    rows = "".join(
        f"<tr><th>{html.escape(m)}</th>"
        + "".join(f"<td>{_fmt(runs[n][sample_id]['metrics'][m])}</td>" for n in present)
        + "</tr>"
        for m in metric_names
    )
    missing = [n for n in runs if n not in present]
    note = (
        f"<p class='warn'>not generated in: {html.escape(', '.join(missing))}</p>"
        if missing
        else ""
    )
    return (
        f"<section id='{anchor}'><h3>{anchor}</h3>"
        f"<p><b>{html.escape(case['group'])}</b> · {html.escape(case['scene_type'])} · "
        f"{html.escape(case['category'])} · seed {instance_seed(sample_id)}</p>"
        f"<p class='prompt'>{html.escape(case['prompt'])}</p>{note}"
        f"<button onclick=\"playRow('{anchor}')\">play row</button>"
        f"<div class='row'>{''.join(figures)}</div>"
        f"<table><tr><th>metric</th>{head}</tr>{rows}</table></section>"
    )


_STYLE = """
body { font-family: sans-serif; margin: 1.5em; }
table { border-collapse: collapse; margin: 0.5em 0 1em; font-size: 0.9em; }
td, th { border: 1px solid #ccc; padding: 2px 6px; text-align: right; }
th:first-child { text-align: left; }
.better { background: #d8f0d8; } .worse { background: #f6d6d6; }
.row { display: flex; flex-wrap: wrap; gap: 6px; }
figure { margin: 0; } video { width: 416px; height: 240px; background: #000; }
figcaption { font-size: 0.85em; text-align: center; }
.prompt { color: #444; max-width: 1200px; } .warn { color: #b00; }
section { border-top: 1px solid #ddd; padding-top: 0.5em; }
"""

_SCRIPT = """
function playRow(id) {
  const videos = document.getElementById(id).querySelectorAll('video');
  videos.forEach(v => { v.pause(); v.currentTime = 0; });
  videos.forEach(v => v.play());
}
"""


def main() -> None:
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    runs = {name: load_run(root) for name, root in args.runs.items()}
    names = list(runs)
    union = {}
    for cases in runs.values():
        for sample_id, case in cases.items():
            union.setdefault(sample_id, case)
    groups = case_groups(union)
    summary = summarize_runs(runs, groups, args.bootstrap)
    media = {
        name: {
            i: encode_case(case, os.path.join(out, "media", name), i) for i, case in cases.items()
        }
        for name, cases in runs.items()
    }

    warnings = [
        f"{name}: gen/manifest.json records a seed other than the instance's for {ids}"
        for name, ids in ((n, seed_mismatches(r)) for n, r in args.runs.items())
        if ids
    ]
    run_list = "".join(
        f"<li><b>{html.escape(n)}</b>: {html.escape(r)} ({len(runs[n])} cases)</li>"
        for n, r in args.runs.items()
    )
    # Cases are listed under their own move only, not again under the combined groups.
    own_groups = {g: ids for g, ids in groups.items() if g in {c["group"] for c in union.values()}}
    toc = "".join(
        f"<li>{html.escape(g)}: "
        + " ".join(f"<a href='#{html.escape(i)}'>{html.escape(i.split('--')[-1])}</a>" for i in ids)
        + "</li>"
        for g, ids in own_groups.items()
    )
    cases_html = "\n".join(
        case_html(i, runs, media, out) for ids in own_groups.values() for i in ids
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>WorldScore runs: {html.escape(' / '.join(names))}</title>
<style>{_STYLE}</style><script>{_SCRIPT}</script></head><body>
<h1>WorldScore runs side by side</h1>
<ul>{run_list}</ul>
{''.join(f"<p class='warn'>{html.escape(w)}</p>" for w in warnings)}
<p>Differences are paired: each run minus <b>{html.escape(names[0])}</b> on the cases both
scored, mean and 95% bootstrap CI over cases ({args.bootstrap} resamples). ↓ lower is better,
↑ higher is better; green / red when the CI excludes 0 (n ≥ 2). With 8 cases per move the
percentile bootstrap is optimistic: read the colours as where to look, not as significance.
Seeds are fixed per case, the same in every run.</p>
<h2>Summary</h2>
{summary_html(summary, names)}
<h2>Cases</h2>
<ul>{toc}</ul>
{cases_html}
</body></html>
"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        payload = {
            "runs": args.runs,
            "summary": summary,
            "cases": {n: {i: c["metrics"] for i, c in cases.items()} for n, cases in runs.items()},
        }
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"wrote {os.path.join(out, 'index.html')}", flush=True)


if __name__ == "__main__":
    main()
