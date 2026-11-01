#!/bin/bash
# One-time setup for the WorldScore benchmark on NCSA Delta, everything under $WORK:
#   $WORK/venvs/WaveFront     minWM env (submit_delta.sh activates it in the job); may be
#                             shared with other checkouts, so it is only installed if missing
#   $WORK/venvs/worldscore    WorldScore env (install_worldscore.sh)
#   $WORK/WorldScore          WorldScore checkout + metric checkpoints
#   $WORK/data/WorldScore-Dataset
#   $WORK/ckpts               Wan 2.1 base + minWM Stage 1 weights; the repo's ckpts/ links here
#   $WORK/.cache              uv/pip/HF/torch caches, incl. the weights the metrics fetch
#                             on first use (delta_env.sh), so jobs need no internet
#
# Run it from a checkout of feature/benchmark that nothing else switches branches in
# (e.g. $WORK/WaveFront-benchmark; the jobs run that checkout's code), on a login node
# from the repo root, in the background (about an hour,
# mostly downloads and compiling DROID-SLAM). Delta caps each user on a login node at
# 16 cores and 37 GB (62 GB hard, no swap); compiles here use 8 jobs. Compute nodes'
# internet access is undocumented, so everything a job needs is fetched here:
#   export WORK=/work/hdd/<project>/$USER
#   nohup bash tools/eval/worldscore/setup_delta.sh > $WORK/setup_worldscore.log 2>&1 &
# Rerunning skips finished steps. Needs no GPU.
set -eo pipefail

: "${WORK:?set WORK, e.g. export WORK=/work/hdd/<project>/\$USER}"
cd "$(dirname "$0")/../../.."
step() { echo "== $(date +%H:%M:%S) $*"; }

source tools/eval/worldscore/delta_env.sh
# Build temporaries (nvcc, pip) off the login node's shared /tmp.
export TMPDIR=$WORK/.tmp
mkdir -p "$TMPDIR" "$WORK/venvs" "$WORK/data" "$WORK/ckpts" "$WORK/outputs"

step "uv"
[ -x "$WORK/venvs/tools/bin/uv" ] || { python3 -m venv "$WORK/venvs/tools" && "$WORK/venvs/tools/bin/pip" install -q uv; }
export PATH=$WORK/venvs/tools/bin:$PATH

step "minWM env"
P=$WORK/venvs/WaveFront/bin/python
# Other checkouts (e.g. training on another branch) may share this env and its editable
# minwm install, so leave a working env alone. Jobs put their own checkout first on
# PYTHONPATH (submit_delta.sh), so they run their code whichever checkout this points at.
if ! "$P" -c "import torch, minwm" 2>/dev/null; then
  [ -x "$P" ] || uv venv -q --python 3.12 "$WORK/venvs/WaveFront"
  uv pip install -q --python "$P" -r requirements/base.txt
  uv pip install -q --python "$P" --no-deps -e .
fi
"$P" -c "import torch, minwm; print('minWM env: torch', torch.__version__)"

step "Wan 2.1 base (18 GB) + minWM Stage 1 weights (6 GB)"
[ -e ckpts ] || ln -s "$WORK/ckpts" ckpts
hf=$WORK/venvs/WaveFront/bin/hf
"$hf" download Wan-AI/Wan2.1-T2V-1.3B --local-dir ckpts/Wan2.1-T2V-1.3B \
  --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" \
  "diffusion_pytorch_model.safetensors" "config.json" >/dev/null
"$hf" download MIN-Lab/minWM --local-dir ckpts --include "Wan21/Action2V/ar_diffusion_tf/*" >/dev/null
ln -sfnT ar_diffusion_tf ckpts/Wan21/Action2V/stage1_ar_tf
du -shL ckpts/Wan2.1-T2V-1.3B ckpts/Wan21/Action2V/stage1_ar_tf

step "WorldScore dataset (1.2 GB)"
if [ ! -d "$WORK/data/WorldScore-Dataset" ]; then
  curl -fsSL --retry 3 -o "$WORK/data/WorldScore-Dataset.zip" \
    https://huggingface.co/datasets/Howieeeee/WorldScore/resolve/main/WorldScore-Dataset.zip
  rm -rf "$WORK/data/.unzip"
  unzip -q "$WORK/data/WorldScore-Dataset.zip" -d "$WORK/data/.unzip"
  mv "$WORK/data/.unzip/WorldScore-Dataset" "$WORK/data/WorldScore-Dataset"
  rm -rf "$WORK/data/.unzip" "$WORK/data/WorldScore-Dataset.zip"
fi

step "WorldScore env"
# torch 2.5.1+cu121 needs a CUDA 12 nvcc. Delta's default cudatoolkit module is
# CUDA 13 and sets CUDA_HOME to it, so swap whenever the current toolkit isn't 12.
cuda_release() { "${CUDA_HOME:+$CUDA_HOME/bin/}nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p'; }
if [ "$(cuda_release)" != 12 ] && type module >/dev/null 2>&1; then
  module unload cudatoolkit 2>/dev/null || true
  module load cuda/12.9
fi
echo "CUDA_HOME=$CUDA_HOME (CUDA $(cuda_release))"
# Delta's PrgEnv sets CC=cc / CXX=CC, Cray wrappers that add -I.../math_libs/include
# (CUDA 13's cuBLAS headers) even with cudatoolkit unloaded; torch passes $CC to nvcc
# as -ccbin. Build the extensions with plain GCC instead (14, supported by CUDA 12.9).
export CC=gcc CXX=g++
WS_ROOT=$WORK/WorldScore WS_VENV=$WORK/venvs/worldscore UV=uv \
  bash tools/eval/worldscore/install_worldscore.sh

step "done"
cat <<EOF
Put this in ~/.bashrc (run.sh derives WS_ROOT, WS_DATASET and WS_PYTHON from WORK):
  export WORK=$WORK
EOF
