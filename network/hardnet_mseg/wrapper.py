"""Two-class logits wrapper for HarDMSEG (DeepLab-style train/eval loops)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hard_mseg import HarDMSEG


class HarDMSEGTwoClass(nn.Module):
    """Single-channel HarDNet-MSEG head as 2-class logits [background, polyp]."""

    def __init__(self, pretrained_backbone: bool = True, backbone_weight_path: str | None = None):
        super().__init__()
        self.core = HarDMSEG(
            channel=32,
            pretrained_backbone=pretrained_backbone,
            backbone_weight_path=backbone_weight_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.core(x)
        if logit.shape[-2:] != x.shape[-2:]:
            logit = F.interpolate(logit, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([-logit, logit], dim=1)
