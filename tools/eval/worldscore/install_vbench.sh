#!/bin/bash
# Build the VBench environment for score_vbench.py and fetch its weights.
#
# A separate Python 3.10 venv: VBench 0.1.5 (PyPI) pins transformers==4.33.2 and
# numpy<2, which clash with both minWM's and WorldScore's envs. torch 2.5.1+cu121 as
# in WorldScore's env (VBench's own setup asks for CUDA 11.x or 12.1). No GPU or
# compiler needed; the weights (DINO ViT-B/16, CLIP ViT-B/32, AMT-S, MUSIQ-SPAQ, about
# 1 GB) go to VBENCH_CACHE_DIR and TORCH_HOME, so jobs need no internet. Rerunning
# skips what is done. setuptools stays below 81: OpenAI CLIP (background consistency)
# imports pkg_resources, which newer setuptools (84.2.1 got pulled in) no longer ship.
#
# Usage:
#   VB_VENV=<venv dir> VBENCH_CACHE_DIR=<cache dir> bash tools/eval/worldscore/install_vbench.sh
#
# Environment:
#   VB_VENV           venv to create; scoring then uses $VB_VENV/bin/python
#   VBENCH_CACHE_DIR  VBench's weight cache (VBench's default is ~/.cache/vbench)
#   VB_TORCH, VB_TORCHVISION, VB_CUDA
#                     torch stack instead of 2.5.1 / 0.20.1 / cu121, e.g. 2.7.0 / 0.22.0 /
#                     cu128 for Blackwell GPUs
#   UV                uv binary (default: uv on PATH)
set -eo pipefail

: "${VB_VENV:?set VB_VENV}" "${VBENCH_CACHE_DIR:?set VBENCH_CACHE_DIR}"
export VBENCH_CACHE_DIR
UV=${UV:-uv}
here=$(cd "$(dirname "$0")" && pwd)
step() { echo "== $(date +%H:%M:%S) $*"; }
command -v wget >/dev/null || { echo "install_vbench.sh: VBench downloads with wget; none on PATH" >&2; exit 1; }

P=$VB_VENV/bin/python
step "venv and packages"
[ -x "$P" ] || "$UV" venv -q --python 3.10 "$VB_VENV"
printf 'torch==%s\ntorchvision==%s\nnumpy<2\n' "${VB_TORCH:-2.5.1}" "${VB_TORCHVISION:-0.20.1}" \
  >"$VB_VENV/constraints.txt"
"$UV" pip install -q --python "$P" -c "$VB_VENV/constraints.txt" \
  --index-url "https://download.pytorch.org/whl/${VB_CUDA:-cu121}" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match torch torchvision vbench==0.1.5 "setuptools<81"

step "weights"
"$P" "$here/score_vbench.py" --prefetch
"$P" -c "import torch, vbench, clip; print('VBench env ok; GPU visible:', torch.cuda.is_available())"
step "done: VB_PYTHON=$P"
