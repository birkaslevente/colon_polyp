# Polip klasszifikációs élő alkalmazás (SZE szakdolgozat)

Élő kamerakép, videófájl vagy képmappa elemzése: osztály, konfidencia és magyarázó hőtérkép.
Az alap modell a DINOv2, ami `onnxruntime`-mal fut, torch és TensorFlow nélkül.

## Tartalom

| Fájl / mappa | Szerep |
|--------------|--------|
| `live_capture_app.py` | GUI, forráskezelés, elemző |
| `polyp_infer.py` | DINOv2 (ONNX) előfeldolgozás, inferencia, Grad-CAM overlay |
| `models/` | `dinov2_polyp.onnx` (gitignore) és `dinov2_polyp.json` (méret, osztályok, küszöb) |
| `requirements_app.txt` | Futtatási függőségek |
| `requirements_legacy.txt` | Opcionális: régi PyTorch / Keras modellek |
| `network/`, `utils/` | Csak a régi modellekhez kell |
| `APP_USAGE_HU.md` | Használat, billentyűk, modellfájlok helye |

## Indítás

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements_app.txt
python live_capture_app.py
```

A `dinov2_polyp.onnx` (kb. 334 MB) nem része a repónak, a `models/` mappába kell másolni.
Klinikai képek és betanított súlyfájlok adatvédelmi és méretkorlát miatt szintén nincsenek a repóban.

## DINOv2 modell

- Bemenet: `float32 [1,3,224,224]`, RGB, 0..1, négyzetes letterbox fekete kitöltéssel. Az ImageNet mean/std a gráfban van.
- Kimenet `probs`: `[1,2]` softmax, az 1-es index a neoplasztikus (`class_1_OTHER`), küszöb 0.5.
- Kimenet `cam`: `[1,2,16,16]` ReLU Grad-CAM mindkét osztályra, a `polyp_infer.py` ebből készít overlayt az eredeti képre.
- Architektúra: Meta DINOv2 ViT-B/14 (ImageNet-SSL), lineáris fej. 5-fold páciens-split CV: BAcc 0.918, AUC 0.957, érzékenység 0.911, specificitás 0.925.

Parancssorból: `python polyp_infer.py models/dinov2_polyp.onnx kep.png --out overlay.png`

## Upstream alap

A régi szegmentáló architektúra (`network/`) a [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) nyílt forráskódú projektre épül (MIT licenc).
