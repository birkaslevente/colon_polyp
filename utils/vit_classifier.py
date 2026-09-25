"""ViT-B/16 polyp binary classifier (full-frame NBI, not ROI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

IMG_SIZE = 224
THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Align with live_capture_app.CLASSIFIER_CLASS_NAMES
CLASS_NAMES = [
    "Non-neoplastic (JNET 1)",
    "Neoplastic (JNET 2a/2b/3)",
]


def build_vit_b16(num_classes: int = 2) -> nn.Module:
    model = vit_b_16(weights=None, image_size=IMG_SIZE)
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    return model


def load_vit_checkpoint(pth_path: str | Path, device: torch.device) -> nn.Module:
    """Build ViT-B/16 and load state_dict-only checkpoint."""
    path = Path(pth_path)
    model = build_vit_b16(num_classes=2)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # not CenterCrop — full frame
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def preprocess_rgb_uint8(img_rgb: np.ndarray) -> torch.Tensor:
    """Full-frame RGB uint8 (H×W×3) → (1, 3, 224, 224). Never use white-bg ROI."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got shape {getattr(img_rgb, 'shape', None)}")
    pil = Image.fromarray(img_rgb.astype(np.uint8)).convert("RGB")
    return _transform()(pil).unsqueeze(0)


@torch.inference_mode()
def predict_vit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    threshold: float = THRESHOLD,
) -> tuple[str, float, float]:
    """
    Softmax p1 (neoplastic) vs threshold.

    Returns: (class_label, confidence, p1_malignant_prob)
    """
    device = next(model.parameters()).device
    x = input_tensor.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    p0, p1 = float(probs[0]), float(probs[1])
    label_idx = int(p1 >= threshold)
    conf = p1 if label_idx == 1 else p0
    return CLASS_NAMES[label_idx], conf, p1
