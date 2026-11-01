"""Download the weights WorldScore's static-split metrics fetch on first use.

Run once on a node with internet (a login node is fine: it only downloads and
builds the models on CPU), in the WorldScore environment, with TORCH_HOME and
HF_HOME pointing where scoring jobs will look (setup_delta.sh sets both under
$WORK). Scoring jobs then need no internet and write no caches to $HOME. Covers:

- VGG19 (style consistency) and ResNet34 (SEA-RAFT's backbone init): torchvision
- CLIP-IQA+ and LAION aesthetic (subjective quality): pyiqa, under TORCH_HOME (the
  latter via openai-clip, redirected there from ~/.cache/clip)
- CLIP ViT-B/16 (content alignment) and BERT (GroundingDINO's text encoder): HF

::

    TORCH_HOME=$WORK/.cache/torch HF_HOME=$WORK/.cache/huggingface \\
        $WORK/venvs/worldscore/bin/python tools/eval/worldscore/prefetch_metric_weights.py
"""

import os

from _common import keep_clip_downloads_off_home


def main() -> None:
    from torchvision.models import ResNet34_Weights, VGG19_Weights, resnet34, vgg19

    vgg19(weights=VGG19_Weights.DEFAULT)
    resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    import pyiqa

    keep_clip_downloads_off_home()

    for name in ("clipiqa+", "laion_aes"):
        pyiqa.create_metric(name, device="cpu")

    from torchmetrics.multimodal.clip_score import CLIPScore
    from transformers import AutoTokenizer, BertModel

    CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    AutoTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")

    import torch

    print(f"metric weights cached: torch hub {torch.hub.get_dir()}, HF {os.getenv('HF_HOME')}")


if __name__ == "__main__":
    main()
