"""Baseline 2 on mock data: the real model, checkpoint and recipe, with random clips.

A first run on Delta before real data exists. It checks the pipeline end to end
(Stage 1 load with the zero-filled disparity branch, training, validation, W&B)
and measures step time, validation time and peak memory. Clips match the planned
45-pixel-frame clips at 480x832: 12 latent frames (3 AR chunks) of 60x104 latents.

    torchrun --nproc_per_node=4 tools/train_mwm.py \
        --config-file configs/wan21/wavefront/train/b2_joint_depth_mock.py \
        --output-dir outputs/smoke-b2-mock
"""

_base_ = "b2_joint_depth.py"

_MOCK = dict(
    _delete_=True,  # replace the inherited LMDB node rather than merge into it
    type="MockCameraLatentDataset",
    num_frames=12,
    channels=16,
    height=60,
    width=104,
    with_disparity=True,
)

training = dict(
    max_steps=20,
    ckpt_interval=0,
    eval_interval=10,
    output_dir="outputs/smoke-b2-mock",
)

data = dict(
    dataset=dict(_MOCK, length=64),
    val_dataset=dict(_MOCK, length=8, seed=10_000),
)

monitor = dict(
    wandb_group="smoke",
    wandb_tags=["b2", "mock"],
    wandb_notes="Baseline 2 on mock data: step time, validation time, peak memory",
)
