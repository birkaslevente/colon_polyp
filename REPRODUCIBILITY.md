# Reprodukálhatósági útmutató

Ez a dokumentum az egyetemi reprodukálhatósági követelmények teljesítéséhez készült.

## Mi van a repóban

- Tanítási notebookok:
  - `colon_short_szd_r.ipynb` (DeepLabV3+)
  - `colon_short_szd_r U-net.ipynb` (U-Net)
- Valós idejű alkalmazás:
  - `live_capture_app.py`
- Függőségek:
  - `requirements_train.txt` (tanítás/notebook)
  - `requirements_app.txt` (alkalmazás futtatás)

## Mi nincs a repóban

Adatvédelmi és méretkorlát miatt ezek nem kerülnek fel:

- klinikai képek és maszkok (`images/`, egyéb betegadatot tartalmazó mappák),
- betanított modellek (`*.pth`, `*.keras`, `*.h5`, `*.pt`, `*.onnx`),
- lokális futási naplók (`performance_log.csv`, `saved_results/`).

## Környezet létrehozása

Javasolt Python verzió: `3.11.x`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

Tanításhoz:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_train.txt
```

App futtatáshoz:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_app.txt
python -m pip install pygrabber
```

## Tanítás reprodukálása

1. Nyisd meg a `colon_short_szd_r.ipynb` notebookot.
2. Állítsd be a lokális adatútvonalakat a saját környezetedre.
3. Futtasd a cellákat sorrendben.
4. Ugyanígy futtasd az `colon_short_szd_r U-net.ipynb` notebookot is.

Megjegyzés: a repóban szereplő notebookok a tanítási folyamatot és a kiértékelést tartalmazzák, de a forrásadatokat nem.

## App futtatása

Az app a következő modelleket várja a projekt gyökerében:

- `checkpoints_0409/` (DeepLab checkpoint: `best_model_0409_epoch_31.pth`)
- `checkpoints_unet/` (U-Net checkpoint)
- `classificator_models/` (`.keras` klasszifikátor)

Részletes modellmásolási séma: `deploy_exe/MODELLOK_MASOLASA.txt`.

Indítás:

```bash
python live_capture_app.py
```

## Korlátok és platform

- A `pygrabber` modul Windows-specifikus (DirectShow kamera-listázás).
- GPU/CPU viselkedés és késleltetés hardverfüggő.
- A klinikai adatok hiánya miatt publikus környezetben csak saját/nyilvános adatforrással reprodukálható a teljes pipeline.
