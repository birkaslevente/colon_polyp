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

- `Start Live Input`: élő forrás indítása.
- `Simulation (Images)`: képmappás teszt.
- `Capture & Analyze` vagy `Space`: pillanatkép elemzés.
- `Model` gomb: DeepLabV3+ / U-Net váltás futás közben.

## 5) Kimenetek

A futás során létrejön:

- `saved_results/` (mentett képek/eredmények),
- `saved_results/predictions.db` (SQLite napló),
- `performance_log.csv` (latencia napló).

## 6) Gyakori hibák

- **Klasszifikátor nem töltődik**: ellenőrizd a `classificator_models/` mappát és a `.keras` fájlt.
- **Nincs kamera**: videófájl vagy szimulációs mód használata.
- **Lassú futás**: CPU-only környezetben várhatóan nagyobb késleltetés.
