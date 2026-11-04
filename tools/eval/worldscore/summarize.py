"""Aggregate WorldScore ``evaluation.json`` files into per-camera-move tables.

Writes ``summary.md`` and ``summary.json`` under the run's output root. Aspect
columns are WorldScore's normalized scores x100, aggregated with its formula
(per-metric mean over instances, then mean over the aspect's metrics); the
overall WorldScore-Static / -Dynamic use its formula too, but over this subset
only, so they are not comparable with the leaderboard (-Dynamic and the dynamic
table only when the run has the dynamic split). Raw columns are means of
the unnormalized metrics, leaving out WorldScore's error sentinel (+-100), whose
count is reported instead::

    python tools/eval/worldscore/summarize.py --runs-root $OUT_DIR
"""

import argparse
import json
import os
from collections import defaultdict
from statistics import mean

from _common import CAMERA_MOVES, instance_dirs

STATIC_ASPECTS = (
    "camera_control",
    "object_control",
    "content_alignment",
    "3d_consistency",
    "photometric_consistency",
    "style_consistency",
    "subjective_quality",
)
DYNAMIC_ASPECTS = ("motion_accuracy", "motion_magnitude", "motion_smoothness")
STATIC_RAW = {
    "rot err (deg)": ("camera_control", "camera_error", 0),
    "trans err": ("camera_control", "camera_error", 1),
    "reproj err": ("3d_consistency", "reprojection_error", None),
    "flow AEPE": ("photometric_consistency", "optical_flow_aepe", None),
}
DYNAMIC_RAW = {
    "motion acc": ("motion_accuracy", "motion_accuracy", None),
    "flow mag": ("motion_magnitude", "optical_flow", None),
}
# Only translation makes parallax; a pan rotates in place.
NO_PARALLAX = {"pan_left", "pan_right"}
_SENTINELS = (100, -100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", required=True, help="this run's output root")
    return parser.parse_args()


def load(runs_root: str, split: str) -> list[tuple[str, dict | None]]:
    """Return ``(group, evaluation or None)`` per prepared instance of a split."""
    rows = []
    for d in instance_dirs(runs_root, split):
        with open(os.path.join(d, "image_data.json"), encoding="utf-8") as f:
            image_data = json.load(f)
        group = image_data["camera_path"][0] if split == "static" else image_data["motion_type"]
        evaluation = None
        if os.path.exists(os.path.join(d, "evaluation.json")):
            with open(os.path.join(d, "evaluation.json"), encoding="utf-8") as f:
                evaluation = json.load(f)
        rows.append((group, evaluation))
    return rows


def aspect_scores(evals: list[dict], aspects: tuple[str, ...]) -> dict[str, float | None]:
    """WorldScore's aspect score x100 over ``evals`` (``None`` if no instance has it)."""
    scores = {}
    for aspect in aspects:
        per_metric = defaultdict(list)
        for ev in evals:
            for metric, score in ev.get(aspect, {}).items():
                if score:
                    per_metric[metric].append(score["score_normalized"])
        scores[aspect] = 100 * mean(mean(v) for v in per_metric.values()) if per_metric else None
    return scores


def raw_means(evals: list[dict], spec: dict) -> tuple[dict[str, float | None], int]:
    """Mean raw metric per column, skipping sentinels; also count the sentinels."""
    means, sentinels = {}, 0
    for column, (aspect, metric, index) in spec.items():
        values = []
        for ev in evals:
            score = ev.get(aspect, {}).get(metric, {}).get("score")
            if score is None:
                continue
            if isinstance(score, (int, float)) and score in _SENTINELS:
                sentinels += 1
                continue
            values.append(score if index is None else score[index])
        means[column] = mean(values) if values else None
    return means, sentinels


def group_rows(
    rows: list[tuple[str, dict | None]], groups: dict[str, set[str]], aspects: tuple, raw: dict
) -> list[dict]:
    """One summary row per named group of instance groups."""
    out = []
    for name, members in groups.items():
        picked = [ev for group, ev in rows if group in members]
        evals = [ev for ev in picked if ev is not None]
        means, sentinels = raw_means(evals, raw)
        out.append(
            {
                "group": name,
                "scored": len(evals),
                "total": len(picked),
                "sentinels": sentinels,
                **aspect_scores(evals, aspects),
                **means,
            }
        )
    return out


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    def cell(value) -> str:
        if value is None:
            return "–"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    static, dynamic = load(args.runs_root, "static"), load(args.runs_root, "dynamic")

    present = [m for m in CAMERA_MOVES if m in {g for g, _ in static}]
    static_groups = {g: {g} for g in present}
    static_groups["有视差（push/pull/move/orbit）"] = set(present) - NO_PARALLAX
    static_groups["无视差（pan）"] = set(present) & NO_PARALLAX
    static_groups["全部"] = set(present)
    dynamic_present = sorted({g for g, _ in dynamic})
    dynamic_groups = {g: {g} for g in dynamic_present}
    dynamic_groups["全部"] = set(dynamic_present)

    static_rows = group_rows(static, static_groups, STATIC_ASPECTS, STATIC_RAW)
    dynamic_rows = group_rows(dynamic, dynamic_groups, DYNAMIC_ASPECTS, DYNAMIC_RAW)

    static_all = aspect_scores([ev for _, ev in static if ev], STATIC_ASPECTS)
    dynamic_all = aspect_scores([ev for _, ev in dynamic if ev], DYNAMIC_ASPECTS)
    # WorldScore counts an aspect with no scored instance as 0.
    static_vals = [static_all[a] or 0.0 for a in STATIC_ASPECTS]
    overall = {"WorldScore-Static": mean(static_vals)}
    if dynamic:
        dynamic_vals = [dynamic_all[a] or 0.0 for a in DYNAMIC_ASPECTS]
        overall["WorldScore-Dynamic"] = mean(static_vals + dynamic_vals)

    common = ["group", "scored", "total", "sentinels"]
    report = [
        f"# WorldScore 子集：{os.path.abspath(args.runs_root)}",
        "",
        "整体（WorldScore 的公式，只在这个子集上算，不能和榜单直接比）："
        + "，".join(f"{k} {v:.2f}" for k, v in overall.items()),
        "",
        "## 静态，按相机运动类型",
        "",
        markdown_table(static_rows, common + list(STATIC_ASPECTS) + list(STATIC_RAW)),
        "",
    ]
    if dynamic:
        report += [
            "## 动态，按运动类型",
            "",
            markdown_table(dynamic_rows, common + list(DYNAMIC_ASPECTS) + list(DYNAMIC_RAW)),
            "",
        ]
    with open(os.path.join(args.runs_root, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    with open(os.path.join(args.runs_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "static": static_rows, "dynamic": dynamic_rows}, f, indent=2)
    print("\n".join(report))


if __name__ == "__main__":
    main()
