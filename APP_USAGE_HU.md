# Alkalmazás használati útmutató (HU)

Ez a dokumentum a `live_capture_app.py` futtatásához ad rövid, gyakorlati lépéseket.

## 1) Előkészületek

- Python: `3.11.x`
- Függőségek:

```bash
python -m pip install -r requirements_app.txt
python -m pip install pygrabber
```

- Modellfájlok a projekt gyökerében:
  - `checkpoints_0409/` (DeepLabV3+)
  - `checkpoints_unet/`
  - `classificator_models/`

Pontos fájlnév- és mappaséma: `deploy_exe/MODELLOK_MASOLASA.txt`.

## 2) Indítás

```bash
python live_capture_app.py
```

## 3) Bemeneti források

Az alkalmazás több forrást kezel:

- kamera (ha elérhető),
- videófájl (`.mp4`, `.avi`, `.mov`, `.mkv`),
- szimulációs mód (`images/` mappa).

## 4) Fő működés

Az ablak induláskor kitölti a munkaterületet, a monitor arányában (16:9, 16:10 vagy 4:3), utána nem méretezhető. A 16:10 ugyanazt az elrendezést használja, mint a 16:9: egy beállítássor, bal oldalon két kisebb kép egymás alatt, jobb oldalon a nagy eredmény.

- `Élő indítás`: élő forrás indítása.
- `Szimuláció`: képmappás teszt.
- `Elemzés (Space)` vagy `Space`: pillanatkép elemzés.
- Az eredményen az osztálynév JNET nélkül jelenik meg (`Nem neoplasztikus` / `Neoplasztikus`), mellette a konfidencia. ResNet úton külön látszik a szegmentáció százaléka is.
- `Szegmentáló` legördülő: DeepLabV3+ / U-Net (**csak ResNet klasszifikátornál** aktív).
- `Modell` legördülő (klasszifikátor):
  - **ViT-B/16** (alap, ha a `vit_final_224.pth` megvan): **csak full-frame klasszifikáció** — **szegmentálás nem fut**. Softmax `p1 ≥ 0.5`.
  - **ResNet50V2**: szegmentálás + **ROI** fehér háttéren (512×512).
- `Magyarázat` menü: ViT attention / ResNet Grad-CAM / szegmentálás Grad-CAM (utóbbi ResNet úton), vagy ki. Az `M` billentyű a kiválasztott magyarázat és a Ki állapot között vált.

## 5) Kimenetek

A futás során létrejön:

- `saved_results/` (mentett képek/eredmények),
- `saved_results/predictions.db` (SQLite napló),
- `performance_log.csv` (latencia napló).

## 6) Gyakori hibák

- **Klasszifikátor nem töltődik**: ellenőrizd a `classificator_models/vit_gui_share/vit_final_224.pth` (ViT) és/vagy a `.keras` fájlt (ResNet).
- **ViT gyenge eredmény**: ne adj croplt / fehér hátteres ROI-t — a modell teljes NBI frame-re tanult.
- **Nincs kamera**: videófájl vagy szimulációs mód használata.
- **Lassú futás**: CPU-only környezetben a ViT (~86 M param) lassabb lehet, mint a Keras ResNet.
