# Polip klasszifikációs élő alkalmazás

Élő kamerakép, videófájl vagy képmappa elemzése: osztály, konfidencia és magyarázó hőtérkép.
Az alap modell a DINOv2, ami `onnxruntime`-mal fut, torch és TensorFlow nélkül.

## Indítás

**Portable zip (ajánlott):** kicsomagolás tetszőleges mappába, majd `start.bat`. Python, DINOv2, DeepLabV3+ és ResNet50V2 benne van, telepítés nem kell. Ha hiányzó DLL-t jelez, kell a Microsoft Visual C++ 2015-2022 x64 Redistributable.

**Forrásból:**

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements_app.txt
python live_capture_app.py
```

A `dinov2_polyp.onnx` (kb. 334 MB) nincs a repóban. A `models/` mappába kell másolni, a `dinov2_polyp.json` mellé.
Más helyről: `DINOV2_MODEL_PATH` környezeti változó (a `.json` az `.onnx` mellett legyen).

## Használat

Források a `Forrás` listában: kamera, videófájl (`.mp4`, `.avi`, `.mov`, `.mkv` az app mappájában). A `Szimuláció` gomb az `images/` mappa képeit játssza le (png, jpg), 2 másodpercenként új képpel.

- `Élő indítás` / `Szimuláció`: a forrás indítása.
- `Élő elemzés`: folyamatos elemzés hőtérkép nélkül.
- `Elemzés (Space)`: pillanatkép elemzése. A `Space` bárhonnan indítja, kivéve ha gomb van fókuszban, ott `Enter` kell.
- `Klasszifikátor`:
  - **DINOv2** (alap), valamint a ViT-B/16, ha a súlyfájl megvan: teljes képes klasszifikáció, szegmentálás nincs.
  - **ResNet50V2**: szegmentálás és ROI klasszifikáció, a `Szegmentáló` menü (DeepLabV3+) aktív.
- `Magyarázat`: DINOv2-nél a gráfba épített Grad-CAM, ViT-nél attention, ResNetnél Grad-CAM. Az `M` billentyű a kiválasztott magyarázat és a `Ki` között vált.
- `Kézi mentés` / `Mentés`: PNG képek és SQLite sor a `saved_results/` mappába, a naplóban a futtatott modellek nevével.
- `Tab` / `Shift+Tab` lépked a vezérlők között, `Enter` aktivál.

Kimenetek: `saved_results/`, `saved_results/predictions.db`, `performance_log.csv`, indítási hiba esetén `LiveCapture_crash.log`.

## DINOv2 modell

- Bemenet: `float32 [1,3,224,224]`, RGB, 0..1, négyzetes letterbox fekete kitöltéssel. Az ImageNet mean/std a gráfban van.
- Kimenet `probs`: `[1,2]` softmax, az 1-es index a neoplasztikus (`class_1_OTHER`), küszöb 0.5.
- Kimenet `cam`: `[1,2,16,16]` ReLU Grad-CAM mindkét osztályra, a `polyp_infer.py` ebből készít overlayt.
- Architektúra: Meta DINOv2 ViT-B/14, lineáris fej. A mellékelt súly az összes adaton tanult (`foldall`), ezért saját, független kiértékelése nincs. Az 5-fold páciens-split CV eredmények (BAcc 0.918, AUC 0.957, érzékenység 0.911, specificitás 0.925) a tanítási jegyzetből származnak, nem ebből az alkalmazásból.
- A Grad-CAM hőtérkép a modellbe van sütve. 24 mintaképen nézve egy patch méretű, a képszélen és a fekete sávban is megjelenő pontokat mutat, nem a polipot, ezért lokalizációra nem használható.
- Egy elemzés CPU-n 0.75 s volt a fejlesztői gépen (mért, gépfüggő).

Parancssorból: `python polyp_infer.py models/dinov2_polyp.onnx kep.png --out overlay.png`

## Régi modellek (opcionális)

DeepLabV3+ szegmentálás (PyTorch) és ResNet50V2 ROI klasszifikátor (Keras). Csomagok: `python -m pip install -r requirements_legacy.txt` (torch CPU, torchvision, tensorflow). Súlyfájlok a projekt gyökerében:

```
checkpoints_0409/best_model_0409_epoch_31.pth
classificator_models/cnn_s2_lr2e4_tb_os.keras      (a *_metadata.json mellé, a legfrissebb *.keras kerül kiválasztásra)
```

Opcionális: U-Net (`checkpoints_unet/best_model_epoch_22.pth`, kell hozzá a `segmentation-models-pytorch`) és ViT-B/16 (`classificator_models/vit_gui_share/vit_final_224.pth`).
Felülírás: `CLASSIFIER_MODEL_PATH` (Keras), `VIT_MODEL_PATH` (ViT). Az app csak a ténylegesen betöltött modelleket kínálja.
Mért futásidő ezen a fejlesztői gépen (CPU): DINOv2 kb. 0.5 s, DeepLabV3+ és ResNet50V2 kb. 1.3 s, a ResNet Grad-CAM kb. 3 s.

## Fájlok

| Fájl / mappa | Szerep |
|--------------|--------|
| `live_capture_app.py` | GUI, forráskezelés, elemző |
| `polyp_infer.py` | DINOv2 (ONNX) előfeldolgozás, inferencia, Grad-CAM overlay |
| `models/` | `dinov2_polyp.onnx` (gitignore) és `dinov2_polyp.json` |
| `network/`, `utils/` | Csak a régi modellekhez kell |
| `requirements_app.txt`, `requirements_legacy.txt` | Függőségek |

## Hibaelhárítás

- **Nincs betöltött modell**: az állapotsor kiírja, hova várja a DINOv2 `.onnx` fájlt.
- **Régi modell nem jelenik meg**: hiányzik a `torch` / `tensorflow`, vagy a fájl nincs a fenti helyen. Az indításkor a konzol kiírja az okot.
- **Szimuláció azt írja, nincs képmappa vagy használható kép**: legyen `images/` mappa az app mellett, benne png vagy jpg.

## Licenc és eredet

A régi szegmentáló architektúra (`network/`) a [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) projektre épül (MIT).
