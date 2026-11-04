"""Fetch the SEA-RAFT checkpoint that WorldScore's flow metrics load but its README omits.

WorldScore's photometric-consistency and motion-magnitude metrics build SEA-RAFT
from ``worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth``,
which is not in WorldScore's download list; without it every instance fails to
score. The SEA-RAFT authors publish these weights on Hugging Face as safetensors,
which store the shared ``bn3`` / ``downsample.1`` BatchNorm once. This restores
the alias, checks the result loads into WorldScore's SEA-RAFT with ``strict=True``
(WorldScore itself loads with ``strict=False``, which would hide a mismatch), and
writes the ``.pth``. Runs in the WorldScore environment; skips if the file exists::

    python tools/eval/worldscore/fetch_searaft.py --worldscore-root $WS_ROOT
"""

import argparse
import os
import sys

REPO_ID = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
CKPT = "worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth"
SEA_RAFT = "worldscore/benchmark/metrics/third_party/SEA-RAFT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worldscore-root", required=True, help="WorldScore checkout")
    args = parser.parse_args()
    os.chdir(args.worldscore_root)
    if os.path.exists(CKPT):
        print(f"{CKPT} exists")
        return

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sys.path.insert(0, SEA_RAFT)
    from core.parser import parse_args
    from core.raft import RAFT

    cfg = argparse.Namespace(cfg=f"{SEA_RAFT}/config/eval/spring-M.json", path=None)
    model = RAFT(parse_args(cfg))
    state = load_file(hf_hub_download(REPO_ID, "model.safetensors"))
    for key in model.state_dict():
        alias = key.replace(".downsample.1.", ".bn3.")
        if key not in state and alias in state:
            state[key] = state[alias]
    model.load_state_dict(state, strict=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save(state, f"{CKPT}.tmp")
    os.replace(f"{CKPT}.tmp", CKPT)
    print(f"wrote {CKPT} ({len(state)} tensors, strict load ok)")


if __name__ == "__main__":
    main()
