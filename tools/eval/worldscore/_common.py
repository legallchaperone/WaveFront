"""Shared constants and WorldScore glue for the WorldScore evaluation bridge.

The bridge never edits the WorldScore checkout: the model is registered
in-process (:func:`register_model`) and its config is built here
(:func:`worldscore_config`) instead of from ``config/model_configs/*.yaml``.
Only numpy is imported at module level, so ``pack.py`` / ``summarize.py`` and the
tests run without WorldScore installed.
"""

import glob
import os
import re
import zlib

import numpy as np

MODEL_NAME = "wavefront"
OUTPUT_DIR = "worldscore_output"
# Must match the minWM infer config: num_frames=20 latent frames -> 77 pixel
# frames, latent_shape (16, 60, 104) -> 832x480, fps=16.
NUM_LATENT_FRAMES = 20
NUM_PIXEL_FRAMES = 4 * (NUM_LATENT_FRAMES - 1) + 1
RESOLUTION = (832, 480)
FPS = 16

_ID_SEP = "--"
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# WorldScore's single-move static camera layouts, in report order.
CAMERA_MOVES = (
    "push_in",
    "pull_out",
    "move_left",
    "move_right",
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
)

# Sign of (position x, position z, forward x) of the last camera, in the first
# camera's OpenCV frame (x right, z forward); 0 means "about zero". An orbit moves
# sideways and slightly forward while turning back toward the target ahead.
_EXPECTED_MOTION = {
    "push_in": (0, 1, 0),
    "pull_out": (0, -1, 0),
    "move_left": (-1, 0, 0),
    "move_right": (1, 0, 0),
    "pan_left": (0, 0, -1),
    "pan_right": (0, 0, 1),
    "orbit_left": (-1, 1, 1),
    "orbit_right": (1, 1, -1),
    "fixed": (0, 0, 0),
}


def instance_id(rel_path: str) -> str:
    """Turn an instance path relative to the output root into a minWM sample id.

    minWM names outputs after a sanitized id; a safe id passes through unchanged,
    so ``pack.py`` can map ``{id}.frames`` back with :func:`instance_rel_path`.

    Args:
        rel_path (str): e.g. ``static/photorealistic/indoor/dining_spaces/000``.

    Returns:
        str: e.g. ``static--photorealistic--indoor--dining_spaces--000``.

    Raises:
        ValueError: if the path holds characters minWM would rewrite, or the
            separator itself.
    """
    parts = rel_path.split(os.sep)
    sample_id = _ID_SEP.join(parts)
    if any(_ID_SEP in p for p in parts) or not _SAFE_ID.fullmatch(sample_id):
        raise ValueError(f"instance path {rel_path!r} cannot round-trip through a minWM id")
    if len(sample_id) > 100:
        raise ValueError(f"id {sample_id!r} exceeds minWM's 100-character filename limit")
    return sample_id


def instance_seed(sample_id: str) -> int:
    """Generation seed of an instance, fixed by its id.

    minWM's default ``seed + position`` would change when a resumed run lists
    only the unfinished instances; a seed tied to the id keeps every instance's
    noise the same across resumes, subsets and the methods being compared.
    """
    return zlib.crc32(sample_id.encode()) & 0x7FFFFFFF


def instance_rel_path(sample_id: str) -> str:
    """Inverse of :func:`instance_id`."""
    return os.path.join(*sample_id.split(_ID_SEP))


def instance_dirs(runs_root: str, visual_movement: str) -> list[str]:
    """List prepared instance directories (those holding ``image_data.json``).

    Args:
        runs_root (str): the run's output root.
        visual_movement (str): ``"static"`` or ``"dynamic"``.

    Returns:
        list[str]: sorted instance directories, following WorldScore's layout
        ``static/<style>/<scene_type>/<category>/<name>`` and
        ``dynamic/<style>/<motion_type>/<name>``.
    """
    depth = {"static": 4, "dynamic": 3}[visual_movement]
    pattern = os.path.join(runs_root, OUTPUT_DIR, visual_movement, *["*"] * depth)
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(pattern, "image_data.json")))


def worldscore_to_minwm_poses(
    cameras_interp: list | np.ndarray, translation_scale: float = 1.0
) -> np.ndarray:
    """Convert WorldScore's ground-truth cameras into minWM's per-latent-frame poses.

    WorldScore stores one camera-to-world matrix per pixel frame, with
    OpenGL/Blender camera axes (x right, y up, z backward). minWM takes one
    world-to-camera matrix per latent frame, with OpenCV axes (x right, y down,
    z forward), the first being identity. Latent frame k takes pixel frame 4k, the
    last pixel frame its VAE group covers.

    Args:
        cameras_interp (list | np.ndarray): ``(NUM_PIXEL_FRAMES, 4, 4)`` c2w matrices,
            the ``cameras_interp`` entry of WorldScore's ``camera_data.json``.
        translation_scale (float): multiplies every translation. WorldScore's
            camera-control metric fits the translation scale, so this only sets
            how far the model is asked to move.

    Returns:
        np.ndarray: ``(NUM_LATENT_FRAMES, 4, 4)`` float32 w2c matrices.

    Raises:
        ValueError: if the camera count does not match ``NUM_PIXEL_FRAMES``.
    """
    c2w = np.asarray(cameras_interp, dtype=np.float64)
    if c2w.shape != (NUM_PIXEL_FRAMES, 4, 4):
        raise ValueError(f"expected ({NUM_PIXEL_FRAMES}, 4, 4) cameras, got {c2w.shape}")
    c2w = c2w @ _GL_TO_CV
    c2w = np.linalg.inv(c2w[0]) @ c2w[4 * np.arange(NUM_LATENT_FRAMES)]
    c2w[:, :3, 3] *= translation_scale
    return np.linalg.inv(c2w).astype(np.float32)


def check_motion(layout: str, w2c: np.ndarray, translation_scale: float = 1.0) -> None:
    """Check that converted poses move the way the WorldScore layout name says.

    A convention slip (axes, c2w vs w2c) raises no error downstream; it only
    makes the model follow a mirrored path. This catches it before generation.

    Args:
        layout (str): WorldScore camera layout, e.g. ``"pan_left"``.
        w2c (np.ndarray): poses from :func:`worldscore_to_minwm_poses`.
        translation_scale (float): the scale passed to that conversion.

    Raises:
        ValueError: if a sign of the final position or heading is wrong.
    """
    c2w = np.linalg.inv(w2c[-1])
    position = c2w[:3, 3] / translation_scale
    observed = (position[0], position[2], c2w[0, 2])
    for name, value, sign in zip(
        ("position x", "position z", "forward x"), observed, _EXPECTED_MOTION[layout]
    ):
        if (sign == 0 and abs(value) > 1e-3) or (sign != 0 and np.sign(value) != sign):
            raise ValueError(f"{layout}: final camera {name}={value:.4f}, expected sign {sign}")


def register_model() -> None:
    """Register :data:`MODEL_NAME` as a WorldScore video model, in this process only."""
    from worldscore.benchmark.utils.modeltype import type2model

    if MODEL_NAME not in type2model["videogen"]:
        type2model["videogen"].append(MODEL_NAME)


def worldscore_config(
    worldscore_root: str, dataset_root: str, runs_root: str, visual_movement: str
) -> dict:
    """Build the config WorldScore would merge from base and model YAMLs.

    Args:
        worldscore_root (str): the WorldScore checkout.
        dataset_root (str): the ``WorldScore-Dataset`` directory.
        runs_root (str): this run's output root; instances go under
            ``<runs_root>/worldscore_output/``.
        visual_movement (str): ``"static"`` or ``"dynamic"``.

    Returns:
        dict: the resolved config.
    """
    from omegaconf import OmegaConf

    base = OmegaConf.load(os.path.join(worldscore_root, "config", "base_config.yaml"))
    model = OmegaConf.create(
        {
            "model": MODEL_NAME,
            "runs_root": runs_root,
            "output_dir": OUTPUT_DIR,
            "benchmark_root": worldscore_root,
            "dataset_root": dataset_root,
            "resolution": list(RESOLUTION),
            "generate_type": "t2v",
            "frames": NUM_PIXEL_FRAMES,
            "fps": FPS,
            "visual_movement": visual_movement,
        }
    )
    return OmegaConf.to_container(OmegaConf.merge(base, model), resolve=True)
