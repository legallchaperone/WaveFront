# Cache and download locations on NCSA Delta, sourced by setup_delta.sh and by the
# jobs submit_delta.sh writes, so both agree. Everything goes under $WORK: Delta's
# $HOME is "not intended as a source/destination for I/O during jobs", and
# /work/hdd is "where I/O from jobs should occur" (Delta user guide, Data Management).
: "${WORK:?set WORK, e.g. export WORK=/work/hdd/<project>/\$USER}"
export UV_CACHE_DIR=$WORK/.cache/uv
export UV_PYTHON_INSTALL_DIR=$WORK/.local/uv-python
export PIP_CACHE_DIR=$WORK/.cache/pip
export HF_HOME=$WORK/.cache/huggingface
# torchvision weights and pyiqa (which keeps its files under torch.hub.get_dir()).
export TORCH_HOME=$WORK/.cache/torch
export XDG_CACHE_HOME=$WORK/.cache
export MPLCONFIGDIR=$WORK/.cache/matplotlib
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
