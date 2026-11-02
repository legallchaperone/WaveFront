"""Baseline 2 (joint depth): Stage 1 teacher forcing plus a jointly denoised disparity channel.

Continues the Wan Action2V Stage 1 checkpoint (released as ``ar_diffusion_tf``,
linked as ``stage1_ar_tf``) with the choices in ``docs/ai-generated/decisions.md``:
dense disparity rides as 16 pixel-unshuffled channels after the 16 latent channels
(``use_disparity``), fed as ``a·(2d − 1)`` by ``DisparityChannel`` and trained with
``JointFlowMatchingLoss`` (λ = 1). ``a = 1`` is set only as ``model.disparity_scale``;
``DisparityChannel``, the validation ``disp_err`` and inference all read it from there.
Noise stays uniform within each AR chunk (s = 0). Optimizer, learning rate and batch
size come from the Stage 1 config, so baseline 1 can match them.

The Stage 1 checkpoint has no disparity branch; ``load_model_weights`` fills it
with zeros, so the run starts from Stage 1's exact latent predictions.

Data: the LMDB samples must also carry ``disparity`` ``[F, 1, 4H, 4W]`` in ``[0, 1]``
on the 2-pixel grid (4× the latent grid per side). The SpatialVID preprocessing that
writes it, and the dataset that reads it, are not in the repo yet; until then use
``b2_joint_depth_mock.py``.

    torchrun --nproc_per_node=4 tools/train_mwm.py \
        --config-file configs/wan21/wavefront/train/b2_joint_depth.py \
        --output-dir outputs/b2-jointdisp-a1-lam1
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
    output_dir="outputs/b2-jointdisp-a1-lam1",
)

checkpoint = dict(pretrained=_STAGE1)

model = dict(use_disparity=True, disparity_scale=1.0)

recipe = dict(
    loss=dict(type="JointFlowMatchingLoss", disp_weight=1.0),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        dict(type="DisparityChannel"),
        dict(type="FlowNoise", uniform_across_frames=False, num_frame_per_block=4),
        dict(type="CleanContextNoiseAug", max_timestep=0),
    ],
)

data = dict(
    dataset=dict(type="CameraLatentLMDBDataset", data_path=_LMDB),
    val_dataset=dict(type="CameraLatentLMDBDataset", data_path=_LMDB_VAL),
)

monitor = dict(
    backends=["wandb"],
    wandb_project="wavefront",
    wandb_group="main-table",
    wandb_tags=["b2", "joint-depth", "a1", "lam1"],
    wandb_notes="Baseline 2: Stage 1 + jointly denoised disparity channel, s=0",
)
