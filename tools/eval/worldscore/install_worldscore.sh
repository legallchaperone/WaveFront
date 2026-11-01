#!/bin/bash
# Build WorldScore's evaluation environment and fetch its metric checkpoints.
#
# Follows WorldScore's README ("Evaluation -> Environment Setup" and "Download
# Evaluation Checkpoints") in a separate Python 3.10 venv (it needs torch 2.5.1 +
# cu121; minWM uses 2.9.1), plus the SEA-RAFT checkpoint its list omits. No GPU
# needed: CUDA extensions are compiled for TORCH_CUDA_ARCH_LIST. Every step skips
# what is already done, so rerunning after a failure resumes. Used by setup_delta.sh
# and colab.ipynb.
#
# Usage:
#   WS_ROOT=<checkout dir> WS_VENV=<venv dir> bash tools/eval/worldscore/install_worldscore.sh
#
# Environment:
#   WS_ROOT               where WorldScore is (or will be) cloned
#   WS_VENV               venv to create; scoring then uses $WS_VENV/bin/python
#   WS_COMMIT             WorldScore commit (default: the one the bridge was written against)
#   CUDA_HOME             CUDA 12 toolkit (default: from the nvcc on PATH)
#   TORCH_CUDA_ARCH_LIST  GPU archs to compile for (default: this node's GPU, else "8.0;8.6",
#                         i.e. A100 and A40/A10)
#   MAX_JOBS              parallel compile jobs (default 8)
#   UV                    uv binary (default: uv on PATH)
set -eo pipefail

: "${WS_ROOT:?set WS_ROOT}" "${WS_VENV:?set WS_VENV}"
WS_COMMIT=${WS_COMMIT:-096c75fc4ada9c7d92e03140c3c1f0c8f383b61f}
UV=${UV:-uv}
here=$(cd "$(dirname "$0")" && pwd)
die() { echo "install_worldscore.sh: $*" >&2; exit 1; }
step() { echo "== $(date +%H:%M:%S) $*"; }

if [ -z "${CUDA_HOME:-}" ]; then
  nvcc=$(command -v nvcc) || die "no nvcc on PATH; set CUDA_HOME to a CUDA 12 toolkit"
  CUDA_HOME=$(dirname "$(dirname "$nvcc")")
fi
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH
cuda_major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
# torch 2.5.1+cu121 builds extensions only with a CUDA 12 nvcc.
[ "$cuda_major" = 12 ] || die "nvcc at $CUDA_HOME is CUDA $cuda_major, need 12 (Delta: module load cuda/12.9)"
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  # On a GPU-less node nvidia-smi may exist and print an error to stdout; keep only "X.Y".
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    grep -E '^[0-9]+\.[0-9]+$' | head -1 || true)
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.6}
fi
arch_re='^[0-9]+\.[0-9]+(\+PTX)?([; ][0-9]+\.[0-9]+(\+PTX)?)*$'
[[ $TORCH_CUDA_ARCH_LIST =~ $arch_re ]] || die "TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST' is not a list like '8.0;8.6'"
# FORCE_CUDA: GroundingDINO only builds its CUDA op when it sees a GPU, unless forced.
export TORCH_CUDA_ARCH_LIST MAX_JOBS=${MAX_JOBS:-8} FORCE_CUDA=1 AM_I_DOCKER=False BUILD_WITH_CUDA=True
step "CUDA $(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p') at $CUDA_HOME; archs $TORCH_CUDA_ARCH_LIST"

P=$WS_VENV/bin/python
C=$WS_VENV/constraints.txt
pip_install() { "$UV" pip install -q --python "$P" -c "$C" "$@"; }

step "WorldScore checkout"
[ -d "$WS_ROOT/.git" ] || git clone -q https://github.com/haoyi-duan/WorldScore.git "$WS_ROOT"
git -C "$WS_ROOT" checkout -q "$WS_COMMIT"
git -C "$WS_ROOT" submodule update -q --init --recursive thirdparty/DROID-SLAM
git -C "$WS_ROOT" submodule update -q --init thirdparty/Grounded-Segment-Anything thirdparty/sam2

step "venv and packages"
[ -x "$P" ] || "$UV" venv -q --python 3.10 "$WS_VENV"
printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\nnumpy<2\nsetuptools<70\n' >"$C"
cu121=https://download.pytorch.org/whl/cu121
pip_install torch torchvision torchaudio --index-url $cu121
pip_install xformers==0.0.28.post3 --index-url $cu121
pip_install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip_install pip setuptools wheel packaging ninja open3d tensorboard scipy opencv-python tqdm \
  matplotlib pyyaml evo gdown yacs loguru einops timm imageio spacy catalogue pyiqa torchmetrics \
  pytorch_lightning cvxpy huggingface_hub safetensors

step "DROID-SLAM (the slow part)"
cd "$WS_ROOT/thirdparty/DROID-SLAM"
if ! "$P" -c "import droid_backends, lietorch" 2>/dev/null; then
  # Its setup.py hardcodes sm_60..sm_86 (no sm_90); compile for our archs only.
  # Start from the pristine file so a rerun never rewrites an already-rewritten one.
  git checkout -q -- setup.py
  "$P" - "$TORCH_CUDA_ARCH_LIST" <<'EOF'
import re
import sys

archs = [major + minor for major, minor in re.findall(r"(\d+)\.(\d+)", sys.argv[1])]
flags = "".join(f"\n                    '-gencode=arch=compute_{a},code=sm_{a}'," for a in archs)
src = open("setup.py").read()
src = re.sub(r"(\s*'-gencode=arch=compute_\d+,code=sm_\d+',?)+", flags, src)
open("setup.py", "w").write(src)
EOF
  grep -o "compute_[0-9]*" setup.py | sort -u | tr '\n' ' '
  echo
  "$P" setup.py install >"$WS_VENV/droid_build.log" 2>&1 || { tail -40 "$WS_VENV/droid_build.log"; die "DROID-SLAM build failed"; }
fi

step "Grounded-SAM, SAM2, VFIMamba"
cd "$WS_ROOT/thirdparty/Grounded-Segment-Anything"
pip_install -e segment_anything
pip_install --no-build-isolation -e GroundingDINO
cd "$WS_ROOT/thirdparty/sam2"
pip_install --no-build-isolation -e .
pip_install --no-build-isolation causal_conv1d==1.5.0.post8 mamba_ssm==2.2.4

step "WorldScore"
pip_install "$WS_ROOT"
"$P" -c "import en_core_web_sm" 2>/dev/null || "$P" -m spacy download -q en_core_web_sm

step "metric checkpoints (about 5 GB)"
K=$WS_ROOT/worldscore/benchmark/metrics/checkpoints
mkdir -p "$K"
cd "$K"
get() {
  local f
  f=$(basename "$1")
  [ -s "$f" ] && return
  curl -fsSL --retry 3 -o "$f.part" "$1"
  mv -f "$f.part" "$f"
}
get https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
get https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
get https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
get https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt
get https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba.pkl
[ -d models ] || { get https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip && unzip -q models.zip; }
# DROID-SLAM's official weights (Google Drive id from its README).
[ -s droid.pth ] || "$WS_VENV/bin/gdown" -q 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O droid.pth
"$P" "$here/fetch_searaft.py" --worldscore-root "$WS_ROOT"
ls -la "$K"

step "import check"
cd "$WS_ROOT"
"$P" -c "import droid_backends, lietorch, groundingdino, sam2, mamba_ssm, worldscore"
# The metrics package imports every metric (and their CUDA ops) at once.
"$P" -c "import torch, worldscore.benchmark.metrics; print('WorldScore env ok; GPU visible:', torch.cuda.is_available())"

step "weights the metrics download on first use (to TORCH_HOME / HF_HOME)"
"$P" "$here/prefetch_metric_weights.py"
step "done: WS_PYTHON=$P"
