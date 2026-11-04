"""Baseline 2 (joint depth) inference: Stage 1 inference plus the disparity channel.

Same sampler, steps, CFG and frame count as the Stage 1 inference config, so baseline 2
is generated exactly like the original minWM and baseline 1. The model node repeats the
two keys ``train/b2_joint_depth.py`` adds (``tests/test_config.py`` keeps them equal).
The noise has 32 channels, the 16 latent channels plus ``DISP_CHANNELS`` disparity
channels; ``LatentNoise(latent_channels=16)`` draws the latent channels first, so for the
same seed baseline 2 starts from the RGB noise the original minWM and baseline 1 start
from. The loop returns the disparity and ``ResultWriter`` saves it as
``{id}.disparity.pt``.

The checkpoint is a training checkpoint exported with ``tools/export_checkpoint.py``::

    bash tools/eval/worldscore/run_small.sh configs/wan21/wavefront/infer/b2_joint_depth.py \\
        inference.checkpoint=<exported model.pt>
"""

_base_ = "../../action2v/infer/stage1_ar_tf.py"

model = dict(use_disparity=True, disparity_scale=1.5)

inference = dict(
    checkpoint="./ckpts/WaveFront/b2_joint_depth/model.pt",
    output_dir="./outputs/wavefront_b2_joint_depth",
    latent_shape=(32, 60, 104),
    input_preprocessors=[
        dict(type="LatentNoise", latent_channels=16),
        dict(type="CameraTrajectory"),
    ],
)
