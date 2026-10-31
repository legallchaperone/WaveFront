#!/bin/bash
# Evaluate one minWM-family checkpoint on the WorldScore subset, end to end:
# pack leftovers -> prepare -> generate (minWM, one process per GPU) -> pack
# -> score (WorldScore, one process per GPU) -> summarize. Every stage skips
# finished work, so rerunning with the same OUT_DIR resumes.
#
# Usage (from the repo root, in the minWM environment):
#   bash tools/eval/worldscore/run.sh <minWM infer config> [dotlist overrides...]
#   WS_STAGE=score bash tools/eval/worldscore/run.sh
#
# Environment:
#   WS_ROOT          WorldScore checkout (pinned commit, see README.md)
#   WS_DATASET       the WorldScore-Dataset directory
#   WS_PYTHON        python of the WorldScore environment (scoring)
#                    (all three default to setup_delta.sh's layout under $WORK when it is set)
#   WS_PREP_PYTHON   python for prepare.py (default WS_PYTHON); it imports WorldScore
#                    from WS_ROOT, so the minWM env works without the WorldScore build
#   WS_STAGE         all (default) | gen (prepare + generate) | score (score + summarize)
#   OUT_DIR          output root (set by submit_delta.sh; default ./local/worldscore)
#   WS_PREPARE_ARGS  extra prepare.py flags, e.g. "--static-per-group 1 --dynamic-per-group 0"
set -eo pipefail

stage=${WS_STAGE:-all}
case $stage in all | gen | score) ;; *) echo "WS_STAGE must be all, gen or score" >&2; exit 2 ;; esac
if [ "$stage" != score ]; then
  [ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
  cfg=$1
  shift
fi
if [ -n "${WORK:-}" ]; then
  : "${WS_ROOT:=$WORK/WorldScore}" "${WS_DATASET:=$WORK/data/WorldScore-Dataset}"
  : "${WS_PYTHON:=$WORK/venvs/worldscore/bin/python}"
fi
: "${WS_ROOT:?set WS_ROOT}" "${WS_DATASET:?set WS_DATASET}"
[ "$stage" = gen ] && [ -n "${WS_PREP_PYTHON:-}" ] || : "${WS_PYTHON:?set WS_PYTHON}"
prep_python=${WS_PREP_PYTHON:-$WS_PYTHON}
out=${OUT_DIR:-./local/worldscore}
gen=$out/gen
here=$(dirname "$0")
ws_args=(--worldscore-root "$WS_ROOT" --dataset-root "$WS_DATASET" --runs-root "$out")

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
else
  # shellcheck disable=SC2207  # GPU indices have no spaces
  gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader))
fi
ngpu=${#gpus[@]}

mkdir -p "$out"
git -C "$WS_ROOT" rev-parse HEAD >"$out/worldscore_commit.txt"
python "$here/pack.py" --gen-dir "$gen" --runs-root "$out"

if [ "$stage" != score ]; then
  # One OUT_DIR = one model: refuse to resume with another config or overrides,
  # which would mix two models' videos in one table. inference.limit may differ.
  gen_args=$(printf '%s\n' "$cfg" "$@" | grep -v '^inference\.limit=' || true)
  if [ -f "$out/gen_args.txt" ]; then
    if [ "$gen_args" != "$(cat "$out/gen_args.txt")" ]; then
      printf '%s was generated with:\n%s\nnot:\n%s\nUse a new OUT_DIR (run name) for another model.\n' \
        "$out" "$(cat "$out/gen_args.txt")" "$gen_args" >&2
      exit 2
    fi
  else
    printf '%s\n' "$gen_args" >"$out/gen_args.txt"
  fi
  # shellcheck disable=SC2086  # WS_PREPARE_ARGS is a flag list
  "$prep_python" "$here/prepare.py" "${ws_args[@]}" --bench "$out/bench.json" ${WS_PREPARE_ARGS:-}
  if python -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1])))' "$out/bench.json"; then
    torchrun --standalone --nproc_per_node="$ngpu" tools/infer_mwm.py --config-file "$cfg" \
      inference.benchmark="$out/bench.json" inference.output_dir="$gen" inference.save_frames=True "$@"
    python "$here/pack.py" --gen-dir "$gen" --runs-root "$out"
  fi
fi
[ "$stage" != gen ] || exit 0

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
