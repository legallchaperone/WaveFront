"""Baseline 2 short run: can the disparity branch learn with its own learning rate?

Baseline 2 trained the zero-init ``disp_embed`` and ``head.disp_head`` at the trunk's
lr of 2e-6. After 3K steps both were rank 1 and about 50x smaller than their RGB
counterparts, and ``val/loss_disp`` sat at 1.0 (the noise variance) at every noise
level: the disparity was never denoised. This run gives the two layers 50x the lr
(1e-4) for 250 steps on one GPU and validates every 50.

Pass: ``val/loss_disp`` clearly below 1.0 and no longer the same at every noise level.

    torchrun --nproc_per_node=1 tools/train_mwm.py \
        --config-file configs/wan21/wavefront/train/b2_joint_depth_lr_probe.py \
        --output-dir outputs/b2-lr-probe
"""

_base_ = "b2_joint_depth.py"

training = dict(
    max_steps=250,
    ckpt_interval=0,
    eval_interval=50,
    output_dir="outputs/b2-lr-probe",
)

recipe = dict(optimizer=dict(lr_scales={"disp_embed.": 50.0, "head.disp_head.": 50.0}))

monitor = dict(
    wandb_group="smoke",
    wandb_tags=["b2", "lr-probe"],
    wandb_notes="Baseline 2 short run: disparity branch at 50x lr, 250 steps, 1 GPU",
)
