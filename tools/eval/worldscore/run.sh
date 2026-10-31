#!/bin/bash
# Evaluate one minWM-family checkpoint on the WorldScore subset, end to end:
# pack leftovers -> prepare -> generate (minWM, one process per GPU) -> pack
# -> score (WorldScore, one process per GPU) -> summarize. Every stage skips
# finished work, so rerunning with the same OUT_DIR resumes.
#
# Usage (from the repo root, in the minWM environment):
#   bash tools/eval/worldscore/run.sh <minWM infer config> [dotlist overrides...]
#
# Environment:
#   WS_ROOT          WorldScore checkout (pinned commit, see README.md)
#   WS_DATASET       the WorldScore-Dataset directory
#   WS_PYTHON        python of the WorldScore environment
#   OUT_DIR          output root (set by run-submit; default ./local/worldscore)
#   WS_PREPARE_ARGS  extra prepare.py flags, e.g. "--static-per-group 1 --dynamic-per-group 1"
set -eo pipefail

[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }
cfg=$1
shift
: "${WS_ROOT:?set WS_ROOT}" "${WS_DATASET:?set WS_DATASET}" "${WS_PYTHON:?set WS_PYTHON}"
out=${OUT_DIR:-./local/worldscore}
gen=$out/gen
here=$(dirname "$0")
ws_args=(--worldscore-root "$WS_ROOT" --dataset-root "$WS_DATASET" --runs-root "$out")

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
ngpu=${#gpus[@]}

mkdir -p "$out"
git -C "$WS_ROOT" rev-parse HEAD >"$out/worldscore_commit.txt"

python "$here/pack.py" --gen-dir "$gen" --runs-root "$out"
# shellcheck disable=SC2086  # WS_PREPARE_ARGS is a flag list
"$WS_PYTHON" "$here/prepare.py" "${ws_args[@]}" --bench "$out/bench.json" ${WS_PREPARE_ARGS:-}

if python -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1])))' "$out/bench.json"; then
  torchrun --standalone --nproc_per_node="$ngpu" tools/infer_mwm.py --config-file "$cfg" \
    inference.benchmark="$out/bench.json" inference.output_dir="$gen" inference.save_frames=True "$@"
  python "$here/pack.py" --gen-dir "$gen" --runs-root "$out"
fi

status=0
pids=()
for i in "${!gpus[@]}"; do
  # pipefail (inherited) makes the subshell fail when evaluate.py does.
  (CUDA_VISIBLE_DEVICES=${gpus[$i]} "$WS_PYTHON" "$here/evaluate.py" "${ws_args[@]}" \
    --shard "$i" --num-shards "$ngpu" 2>&1 | tee "$out/eval-shard$i.log" | sed -u "s/^/[shard $i] /") &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[ "$status" = 0 ] || echo "some instances failed to score; see $out/eval-shard*.log" >&2

python "$here/summarize.py" --runs-root "$out"
exit "$status"
