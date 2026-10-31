#!/bin/bash
# One-time setup for the WorldScore benchmark on NCSA Delta, everything under $WORK:
#   $WORK/venvs/WaveFront     minWM env (submit_delta.sh activates it in the job)
#   $WORK/venvs/worldscore    WorldScore env (install_worldscore.sh)
#   $WORK/WorldScore          WorldScore checkout + metric checkpoints
#   $WORK/data/WorldScore-Dataset
#   $WORK/ckpts               Wan 2.1 base + minWM Stage 1 weights; the repo's ckpts/ links here
#
# Run it on a login node from the repo root, in the background (about an hour,
# mostly downloads and compiling DROID-SLAM; NCSA's guides put software installs
# and file transfers on login nodes, and compute nodes may lack internet):
#   export WORK=/work/hdd/<project>/$USER
#   nohup bash tools/eval/worldscore/setup_delta.sh > $WORK/setup_worldscore.log 2>&1 &
# Rerunning skips finished steps. Needs no GPU.
set -eo pipefail

: "${WORK:?set WORK, e.g. export WORK=/work/hdd/<project>/\$USER}"
cd "$(dirname "$0")/../../.."
step() { echo "== $(date +%H:%M:%S) $*"; }

export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$WORK/.local/uv-python}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface}
mkdir -p "$WORK/venvs" "$WORK/data" "$WORK/ckpts" "$WORK/outputs"

step "uv"
[ -x "$WORK/venvs/tools/bin/uv" ] || { python3 -m venv "$WORK/venvs/tools" && "$WORK/venvs/tools/bin/pip" install -q uv; }
export PATH=$WORK/venvs/tools/bin:$PATH

step "minWM env"
P=$WORK/venvs/WaveFront/bin/python
[ -x "$P" ] || uv venv -q --python 3.12 "$WORK/venvs/WaveFront"
uv pip install -q --python "$P" -r requirements/base.txt
uv pip install -q --python "$P" --no-deps -e .
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
# Delta's default cudatoolkit is CUDA 13; torch 2.5.1+cu121 needs a CUDA 12 nvcc.
if [ -z "${CUDA_HOME:-}" ] && type module >/dev/null 2>&1; then
  module unload cudatoolkit 2>/dev/null || true
  module load cuda/12.9
fi
WS_ROOT=$WORK/WorldScore WS_VENV=$WORK/venvs/worldscore UV=uv \
  bash tools/eval/worldscore/install_worldscore.sh

step "done"
cat <<EOF
Put this in ~/.bashrc (run.sh derives WS_ROOT, WS_DATASET and WS_PYTHON from WORK):
  export WORK=$WORK
EOF
