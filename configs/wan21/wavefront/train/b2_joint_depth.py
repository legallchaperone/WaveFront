"""Baseline 2 (joint depth): Stage 1 teacher forcing plus a jointly denoised disparity channel.

Continues the Wan Action2V Stage 1 checkpoint (released as ``ar_diffusion_tf``,
linked as ``stage1_ar_tf``) with the choices in ``docs/ai-generated/decisions.md``:
dense disparity rides as 16 pixel-unshuffled channels after the 16 latent channels
(``use_disparity``), fed as ``a·(2d − 1)`` by ``DisparityChannel`` and trained with
``JointFlowMatchingLoss`` (λ = 1). ``a = 1.5`` is set only as ``model.disparity_scale``;
``DisparityChannel``, the validation ``disp_err`` and inference all read it from there.
Noise stays uniform within each AR chunk (s = 0). The checkpoint, main data, budget
and validation are shared with baseline 1 in ``_stage1_continue.py``.

The Stage 1 checkpoint has no disparity branch; ``load_model_weights`` fills it
with zeros, so the run starts from Stage 1's exact latent predictions.

Data: the main SpatialVID LMDBs of ``_stage1_continue.py`` joined by clip id with the
disparity LMDBs written by ``tools/data/wan21/build_spatialvid_disparity_lmdb.py``
(``CameraDisparityLMDBDataset``), which add ``disparity`` ``[F, 1, 4H, 4W]`` in
``[0, 1]`` on the 2-pixel grid (4× the latent grid per side). Clips whose depth failed
the preprocessing checks have no label, so baseline 2 skips them; baseline 1 keeps them.

    torchrun --nproc_per_node=4 tools/train_mwm.py \
        --config-file configs/wan21/wavefront/train/b2_joint_depth.py \
        --output-dir outputs/b2-jointdisp-a1-lam1
"""

_base_ = "_stage1_continue.py"

training = dict(output_dir="outputs/b2-jointdisp-a1-lam1")

model = dict(use_disparity=True, disparity_scale=1.5)

data = dict(
    dataset=dict(
        type="CameraDisparityLMDBDataset",
        disparity_path="./dataset/WaveFront/spatialvid/train_disparity",
    ),
    val_dataset=dict(
        type="CameraDisparityLMDBDataset",
        disparity_path="./dataset/WaveFront/spatialvid/val_disparity",
    ),
)

recipe = dict(
    loss=dict(type="JointFlowMatchingLoss", disp_weight=1.0),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        dict(type="DisparityChannel"),
        dict(type="FlowNoise", uniform_across_frames=False, num_frame_per_block=4),
        dict(type="CleanContextNoiseAug", max_timestep=0),
    ],
)

monitor = dict(
    wandb_tags=["b2", "joint-depth", "a1", "lam1"],
    wandb_notes="Baseline 2: Stage 1 + jointly denoised disparity channel, s=0",
)
