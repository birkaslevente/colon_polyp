"""Polip-klasszifikátor (DINOv2, ONNX) - torch nélküli inferencia + Grad-CAM overlay.

Függőségek: numpy, Pillow, onnxruntime.  (Nincs torch / transformers / matplotlib / cv2.)

Példa:
    clf = PolypClassifier("dinov2_polyp.onnx")          # a mellette lévő dinov2_polyp.json-t is betölti
    res = clf.predict(Image.open("kep.png"))
    res.label, res.prob_neoplastic, res.threshold
    res.overlay        # PIL RGB: az EREDETI kép + Grad-CAM (letterbox-padding nélkül)
    res.heatmap        # float32 [H,W] 0..1, az eredeti kép koordinátáiban
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

LETTERBOX = 448  # az adat-előkészítés négyzet-mérete (resize_letterbox_448.py), a tanításnál is ez volt


def letterbox_to_square(img: Image.Image, size: int = LETTERBOX, fill=(0, 0, 0)):
    """Arányt tartó átméretezés négyzetre, fekete kitöltéssel. Visszaadja a képet és a tartalom-téglalapot."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    x0, y0 = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (x0, y0))
    return canvas, (x0, y0, x0 + new_w, y0 + new_h)


def jet(x: np.ndarray) -> np.ndarray:
    """x in [0,1] -> RGB uint8 (matplotlib 'jet' közelítés)."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


@dataclass
class Result:
    label: str
    class_index: int
    prob_neoplastic: float
    probs: tuple
    threshold: float
    heatmap: np.ndarray          # [H,W] 0..1, az eredeti kép koordinátáiban
    overlay: Image.Image         # eredeti kép + jet overlay
    cam_grid: np.ndarray         # nyers [G,G] CAM a döntött osztályra (normalizálatlan)


class PolypClassifier:
    def __init__(self, onnx_path: str | Path, meta_path: str | Path | None = None, threads: int | None = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        meta_path = Path(meta_path) if meta_path else onnx_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text())
        self.size = int(self.meta["img_size"])
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.class_names = self.meta["class_names"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    # -- előfeldolgozás: pontosan a tanítási pipeline (letterbox 448 LANCZOS -> Resize(S) BILINEAR)
    def preprocess(self, img: Image.Image):
        sq, box = letterbox_to_square(img)
        if self.size != LETTERBOX:
            sq = sq.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = np.asarray(sq, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(x), box

    def predict(self, img: Image.Image, alpha: float = 0.45, cam_class: int | None = None) -> Result:
        img = img.convert("RGB")
        x, box = self.preprocess(img)
        probs, cam = self.sess.run(None, {"image": x})
        probs, cam = probs[0], cam[0]                       # [2], [2,G,G]
        p1 = float(probs[1])
        k = int(p1 >= self.threshold) if cam_class is None else cam_class
        grid = cam[k]
        h = grid - grid.min()
        h = h / h.max() if h.max() > 0 else h
        # G x G -> letterbox négyzet (LETTERBOX) -> kivágás az eredeti tartalomra -> eredeti méret
        sq = Image.fromarray((h * 255).astype(np.uint8)).resize((LETTERBOX, LETTERBOX), Image.Resampling.BICUBIC)
        x0, y0, x1, y1 = box
        crop = sq.crop((x0, y0, x1, y1)).resize(img.size, Image.Resampling.BICUBIC)
        heat = np.asarray(crop, dtype=np.float32) / 255.0
        col = jet(heat)
        base = np.asarray(img, dtype=np.float32)
        over = ((1 - alpha) * base + alpha * col).clip(0, 255).astype(np.uint8)
        return Result(
            label=self.class_names[k if cam_class is not None else int(p1 >= self.threshold)],
            class_index=int(p1 >= self.threshold),
            prob_neoplastic=p1,
            probs=(float(probs[0]), p1),
            threshold=self.threshold,
            heatmap=heat,
            overlay=Image.fromarray(over),
            cam_grid=grid,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--out", default="overlay.png")
    a = ap.parse_args()
    r = PolypClassifier(a.model).predict(Image.open(a.image))
    print(f"{r.label}  p(neoplasztikus)={r.prob_neoplastic:.3f}  küszöb={r.threshold}")
    r.overlay.save(a.out)
    print("overlay:", a.out)
