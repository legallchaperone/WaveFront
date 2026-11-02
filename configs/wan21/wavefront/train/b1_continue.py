"""Baseline 1: Stage 1 continued on the same data, with no changes.

Shows how much the new data and the extra steps alone buy. It shares the
checkpoint, data, budget and validation with baseline 2 (``_stage1_continue.py``)
and keeps the Stage 1 model, loss and preprocessing unchanged.

    torchrun --nproc_per_node=4 tools/train_mwm.py \
        --config-file configs/wan21/wavefront/train/b1_continue.py \
        --output-dir outputs/b1-continue
"""

_base_ = "_stage1_continue.py"

training = dict(output_dir="outputs/b1-continue")

monitor = dict(
    wandb_tags=["b1", "continue"],
    wandb_notes="Baseline 1: Stage 1 continued on the same data, no changes",
)
