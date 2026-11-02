"""Shared setup for the main-table runs that continue Stage 1 (baselines 1 and 2).

Everything those runs must share lives here: the starting checkpoint, the data,
the training budget, the validation cadence and the W&B group. The model,
optimizer, learning rate and batch size come from the Stage 1 config. Each run
adds only what it changes, plus its output dir and W&B tags.

Data: SpatialVID LMDBs, not in the repo yet. Baseline 2 also reads a
``disparity`` key per sample; baseline 1 ignores it.
"""

_base_ = "../../action2v/train/stage1_ar_tf.py"

_STAGE1 = "./ckpts/Wan21/Action2V/stage1_ar_tf/model.pt"
_LMDB = "./dataset/WaveFront/spatialvid/train"
_LMDB_VAL = "./dataset/WaveFront/spatialvid/val"

training = dict(
    max_steps=3_000,
    log_interval=1,
    ckpt_interval=1_000,
    eval_interval=250,
)

checkpoint = dict(pretrained=_STAGE1)

data = dict(
    dataset=dict(type="CameraLatentLMDBDataset", data_path=_LMDB),
    val_dataset=dict(type="CameraLatentLMDBDataset", data_path=_LMDB_VAL),
)

monitor = dict(
    backends=["wandb"],
    wandb_project="wavefront",
    wandb_group="main-table",
)
