#!/bin/bash
# The small WorldScore benchmark: 16 static instances, photorealistic, one camera
# move each -- push_in, move_left, orbit_left (parallax) and pan_left (none) x
# indoor/outdoor x 2 -- and no dynamic split. Enough to watch videos and spot
# gross failures; too few per move to compare methods (use run.sh's full subset).
#
# A thin wrapper over run.sh that fixes the subset, so every small-benchmark run
# evaluates the same 16 instances. The subset nests in the full one: rerunning
# run.sh with the same OUT_DIR grows it without regenerating these 16.
#
# Usage (from the repo root, in the minWM environment):
#   bash tools/eval/worldscore/run_small.sh [minWM infer config] [dotlist overrides...]
#
# The config defaults to the Stage 1 teacher-forcing model (50-step UniPC, CFG 3),
# the baseline WaveFront continues from. Environment as in run.sh (WS_ROOT,
# WS_DATASET, WS_PYTHON, WS_PREP_PYTHON, WS_STAGE); OUT_DIR defaults to
# ./local/worldscore-small. WS_PREPARE_ARGS is set here and must not be.
#
# Delta (one A40, about 2-3 h estimated, unmeasured):
#   bash tools/eval/worldscore/submit_delta.sh <run name>
# Colab: tools/eval/worldscore/colab.ipynb calls this script.
set -eo pipefail

if [ -n "${WS_PREPARE_ARGS:-}" ]; then
  echo "run_small.sh fixes the subset; unset WS_PREPARE_ARGS (or call run.sh directly)" >&2
  exit 2
fi
cfg=configs/wan21/action2v/infer/stage1_ar_tf.py
if [ $# -ge 1 ] && [[ $1 != *=* ]]; then
  cfg=$1
  shift
fi

export WS_PREPARE_ARGS="--moves push_in,move_left,orbit_left,pan_left --static-per-group 2 --dynamic-per-group 0"
export OUT_DIR=${OUT_DIR:-./local/worldscore-small}
exec bash "$(dirname "$0")/run.sh" "$cfg" "$@"
