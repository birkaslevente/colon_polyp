# Alkalmazás használati útmutató (HU)

## 1) Előkészületek

- Függőségek: `python -m pip install -r requirements_app.txt` (torch és TensorFlow nélkül).
- DINOv2 modell: `models/dinov2_polyp.onnx` és `models/dinov2_polyp.json`. Más helyről: környezeti változó `DINOV2_MODEL_PATH` (a `.json` az `.onnx` mellett legyen).
- Opcionális, régi modellek: `python -m pip install -r requirements_legacy.txt`, majd a fájlok a projekt gyökerében:

```
checkpoints_0409/best_model_0409_epoch_31.pth
checkpoints_unet/best_model_epoch_22.pth
classificator_models/cnn_s2_lr2e4_tb_os.keras   (vagy bármely *.keras, a legfrissebb kerül kiválasztásra)
classificator_models/vit_gui_share/vit_final_224.pth
```

A Keras útvonal felülírása: `CLASSIFIER_MODEL_PATH`, a ViT-é: `VIT_MODEL_PATH`.
Az app csak azt a modellt kínálja a menüben, ami ténylegesen betöltődött.

## 2) Indítás

```bash
python live_capture_app.py
```

## 3) Bemeneti források

- kamera (a `Forrás` listában),
- videófájl (`.mp4`, `.avi`, `.mov`, `.mkv` az app mappájában),
- szimuláció: az `images/` mappa képei.

## 4) Kezelés

- `Élő indítás` / `Szimuláció`: a kiválasztott forrás indítása.
- `Élő elemzés`: folyamatos elemzés a képkockákon, hőtérkép nélkül.
- `Elemzés (Space)`: pillanatkép elemzése.
- `Klasszifikátor`:
  - **DINOv2** (alap) és **ViT-B/16**: teljes képes klasszifikáció, szegmentálás nincs. A `Szegmentáló` menü ilyenkor tiltott, a magyarázat csak `Klasszifikáció` vagy `Ki`.
  - **ResNet50V2**: szegmentálás és ROI klasszifikáció, a `Szegmentáló` menü (DeepLabV3+ / U-Net) aktív.
- `Magyarázat`: DINOv2-nél a gráfba épített Grad-CAM, ViT-nél attention, ResNetnél Grad-CAM. Az `M` billentyű a kiválasztott magyarázat és a `Ki` között vált.
- `Kézi mentés` / `Auto mentés`, `Mentés`: PNG képek és SQLite sor a `saved_results/` mappába. A napló rögzíti, melyik modellek futottak.

Billentyűzet: `Tab` / `Shift+Tab` lépked a vezérlők között, `Enter` aktivál, `Space` bárhonnan elemzést indít (ezért fókuszált vezérlőn sem kapcsol, ott `Enter` kell).

## 5) Kimenetek

- `saved_results/` (mentett képek),
- `saved_results/predictions.db` (SQLite napló),
- `performance_log.csv` (késleltetés napló),
- `LiveCapture_crash.log` (indítási hiba esetén).

## 6) Gyakori hibák

- **Nincs betöltött modell**: az állapotsor kiírja, hova várja a DINOv2 `.onnx` fájlt.
- **Régi modell nem jelenik meg a menüben**: hiányzik a `torch` / `tensorflow` (`requirements_legacy.txt`), vagy a fájl a fenti helyen.
- **Nincs kamera**: videófájl vagy szimulációs mód használata. Kameranevekhez kell a `pygrabber` (a `requirements_app.txt` Windowson telepíti).
- **Lassú futás**: a DINOv2 CPU-n fut, a Grad-CAM-mel együtt egy elemzés 0.75 s volt a fejlesztői gépen (mért).
