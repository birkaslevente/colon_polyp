"""Grad-CAM explainability helpers for segmentation and classification overlays."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tensorflow as tf
except Exception:
    tf = None


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ReLU + min-max normalization to [0, 1]."""
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.maximum(cam, 0.0)
    if cam.size == 0:
        return cam
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cam_min) / (cam_max - cam_min)


def apply_colormap(cam_01: np.ndarray) -> np.ndarray:
    """Apply JET colormap; returns RGB uint8."""
    cam_u8 = (np.clip(cam_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def blend_heatmap(base_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB image."""
    base = np.asarray(base_rgb, dtype=np.float32)
    if base.max() > 1.5:
        base = base / 255.0

    hm = np.clip(np.asarray(heatmap_01, dtype=np.float32), 0.0, 1.0)
    colored = apply_colormap(hm).astype(np.float32) / 255.0

    blended = base.copy()
    active = hm > 0.01
    for channel in range(3):
        blended[..., channel] = np.where(
            active,
            (1.0 - alpha) * base[..., channel] + alpha * colored[..., channel],
            base[..., channel],
        )
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


def map_cam_to_full_frame(
    cam_small: np.ndarray,
    bbox: tuple[int, int, int, int],
    out_hw: tuple[int, int] = (513, 513),
) -> np.ndarray:
    """Map a small CAM into the full frame at bbox (rmin, rmax, cmin, cmax)."""
    rmin, rmax, cmin, cmax = bbox
    out_h, out_w = out_hw
    out = np.zeros((out_h, out_w), dtype=np.float32)

    roi_h = int(rmax - rmin + 1)
    roi_w = int(cmax - cmin + 1)
    if roi_h <= 0 or roi_w <= 0:
        return out

    cam_resized = cv2.resize(
        np.asarray(cam_small, dtype=np.float32),
        (roi_w, roi_h),
        interpolation=cv2.INTER_LINEAR,
    )
    out[rmin : rmax + 1, cmin : cmax + 1] = cam_resized
    return out


def find_last_keras_conv_layer(model):
    """Return the last Conv2D layer in the model tree (skips preprocess-only layers)."""
    if tf is None:
        return None
    last = None

    def walk(container):
        nonlocal last
        for layer in container.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last = layer
            if hasattr(layer, "layers") and layer.layers:
                walk(layer)

    walk(model)
    return last


def find_last_keras_conv_layer_name(model) -> str | None:
    """Backward-compatible name helper; may be None for nested Conv2D layers."""
    layer = find_last_keras_conv_layer(model)
    return layer.name if layer is not None else None


def _keras_input_spatial_shape(model):
    """Extract (H, W, C) from model.input_shape across Keras versions."""
    in_shape = model.input_shape
    if isinstance(in_shape, dict):
        in_shape = next(iter(in_shape.values()))
    if isinstance(in_shape, (list, tuple)) and in_shape and isinstance(in_shape[0], (list, tuple)):
        in_shape = in_shape[0]
    if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
        return tuple(in_shape[1:])
    return (512, 512, 3)


def build_keras_grad_model(model, conv_layer):
    """Inference-only path: preprocess → backbone → head (skips random augment layers).

    Fresh Dense layers copy weights from the classifier head to avoid Keras 3
    layer-reuse shape bugs when rebuilding a separate Grad-CAM graph.
    """
    spatial = _keras_input_spatial_shape(model)
    inp = tf.keras.Input(shape=spatial, name="heatmap_input")
    try:
        x = model.get_layer("preprocess_input")(inp)
    except Exception:
        x = tf.keras.applications.resnet_v2.preprocess_input(inp)

    backbone = model.get_layer("resnet50v2")
    backbone_in = backbone.input
    if isinstance(backbone_in, (list, tuple)):
        backbone_in = backbone_in[0]
    backbone_out = backbone.output
    if isinstance(backbone_out, (list, tuple)):
        backbone_out = backbone_out[0]
    conv_out = conv_layer.output
    if isinstance(conv_out, (list, tuple)):
        conv_out = conv_out[0]

    extractor = tf.keras.Model(
        inputs=backbone_in,
        outputs=[conv_out, backbone_out],
        name="cam_extractor",
    )
    conv_maps, features = extractor(x)
    if isinstance(features, (list, tuple)):
        features = features[0]
    if isinstance(conv_maps, (list, tuple)):
        conv_maps = conv_maps[0]

    d256 = model.get_layer("dense_256")
    out = model.get_layer("output")
    units_h = int(d256.units) if not isinstance(d256.units, (list, tuple)) else int(d256.units[-1])
    units_o = int(out.units) if not isinstance(out.units, (list, tuple)) else int(out.units[-1])
    h = tf.keras.layers.Dense(
        units_h,
        activation=d256.activation,
        name="cam_dense_256",
    )(features)
    pred = tf.keras.layers.Dense(
        units_o,
        activation=out.activation,
        name="cam_output",
    )(h)

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_maps, pred], name="grad_cam_model")
    grad_model.get_layer("cam_dense_256").set_weights(d256.get_weights())
    grad_model.get_layer("cam_output").set_weights(out.get_weights())
    return grad_model


def grad_cam_keras(
    model,
    roi_batch: np.ndarray,
    class_index: int,
    conv_layer,
    grad_model=None,
) -> np.ndarray:
    """Grad-CAM for Keras classifier on ROI batch (1, H, W, 3)."""
    if tf is None:
        raise RuntimeError("TensorFlow is not available")

    if isinstance(conv_layer, str):
        conv_layer = model.get_layer(conv_layer)

    if grad_model is None:
        grad_model = build_keras_grad_model(model, conv_layer)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(roi_batch)
        if predictions.shape[-1] == 1:
            if class_index == 1:
                loss = predictions[:, 0]
            else:
                loss = 1.0 - predictions[:, 0]
        else:
            loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        h, w = roi_batch.shape[1], roi_batch.shape[2]
        return np.zeros((h, w), dtype=np.float32)

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    return normalize_cam(cam)


def _get_torch_cam_layer(model: nn.Module, model_type: str) -> nn.Module:
    if model_type == "deeplab":
        return model.classifier.classifier[0]
    if model_type == "unet":
        last_conv = None
        for module in model.segmentation_head.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d found in U-Net segmentation head")
        return last_conv
    raise ValueError(f"Unsupported model_type for Grad-CAM: {model_type}")


def grad_cam_torch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_type: str,
    target_class: int = 1,
) -> np.ndarray:
    """Grad-CAM for PyTorch segmentation model; returns H×W CAM in [0,1]."""
    model.eval()
    layer = _get_torch_cam_layer(model, model_type)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        inp = input_tensor.detach()
        if inp.device != next(model.parameters()).device:
            inp = inp.to(next(model.parameters()).device)
        inp = inp.requires_grad_(True)

        with torch.enable_grad():
            output = model(inp)
            if model_type == "unet":
                score = torch.sigmoid(output).sum()
            else:
                score = output[:, target_class, :, :].sum()
            score.backward()

        if not activations or not gradients:
            h, w = inp.shape[2], inp.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        acts = activations[0][0]
        grads = gradients[0][0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam).detach().cpu().numpy()
        cam = normalize_cam(cam)

        th, tw = inp.shape[2], inp.shape[3]
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_up = F.interpolate(cam_t, size=(th, tw), mode="bilinear", align_corners=False)
        return cam_up.squeeze().numpy()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


# JET: 0.25 már világoskék. Alatta sötétkék, azt nem rajzoljuk ki.
VIT_ATTENTION_FLOOR = 0.25


def vit_attention_map(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """CLS→patch attention from the last ViT encoder block; returns 224×224 map in [0,1].

    Not Grad-CAM — mean over attention heads. Patch grid is 14×14 for ViT-B/16 @ 224.
    Values below VIT_ATTENTION_FLOOR are zero so the overlay starts at light blue.
    """
    model.eval()
    device = next(model.parameters()).device
    inp = input_tensor.detach().to(device)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)

    last_block = model.encoder.layers[-1]
    mha = last_block.self_attention
    captured: list[torch.Tensor] = []
    original_forward = mha.forward

    def patched_forward(query, key, value, *args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        out, weights = original_forward(query, key, value, *args, **kwargs)
        if weights is not None:
            captured.append(weights.detach())
        return out, weights

    mha.forward = patched_forward
    try:
        with torch.inference_mode():
            _ = model(inp)
    finally:
        mha.forward = original_forward

    if not captured:
        return np.zeros((224, 224), dtype=np.float32)

    # weights: (B, num_heads, seq, seq) — CLS at index 0, patches 1..N
    w = captured[-1]
    if w.dim() == 3:
        # (B, seq, seq) already averaged
        cls_to_patch = w[0, 0, 1:]
    else:
        cls_to_patch = w[0, :, 0, 1:].mean(dim=0)

    n_patch = int(cls_to_patch.numel())
    side = int(round(n_patch ** 0.5))
    if side * side != n_patch:
        # Unexpected layout — fall back to flat resize
        cam = cls_to_patch.float().cpu().numpy()
        cam = normalize_cam(cam)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return np.where(cam >= VIT_ATTENTION_FLOOR, cam, 0.0).astype(np.float32)

    cam = cls_to_patch.float().reshape(side, side).cpu().numpy()
    cam = normalize_cam(cam)
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    return np.where(cam >= VIT_ATTENTION_FLOOR, cam, 0.0).astype(np.float32)
