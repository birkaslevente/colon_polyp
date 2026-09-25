# Polip szegmentációs rendszer (SZE szakdolgozat)

Ez a repó a szakdolgozatban használt fejlesztési környezetet tartalmazza:

- notebookos tanítás (`colon_short_szd_r.ipynb`, `colon_short_szd_r U-net.ipynb`),
- valós idejű alkalmazás (`live_capture_app.py`),
- reprodukálhatósági és használati dokumentáció.

## Gyors áttekintés

| Dokumentum / fájl | Tartalom |
|-------------------|----------|
| `requirements_train.txt` | Tanítási függőségek |
| `requirements_app.txt` | Alkalmazás függőségek |
| `REPRODUCIBILITY.md` | Reprodukálhatóság, környezet, seed |
| `APP_USAGE_HU.md` | Élő alkalmazás használata (források, gombok, mentés) |
| `deploy_exe/README_HU.md` | PyInstaller build, másik gépen futtatás |
| `deploy_exe/MODELLOK_MASOLASA.txt` | Betanított súlyok másolása exe mellé |
| `scripts/README.md` | Segédscriptek / tesztek / train (almappák) |

## Fontos korlát

Klinikai képek, maszkok és betanított súlyfájlok adatvédelmi és méretkorlát miatt **nem** részei a publikus repónak. A teljes futtatáshoz saját vagy engedélyezett adatkészlet és modellek szükségesek.

## Minimális indítás (fejlesztői környezet)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
python -m pip install -r requirements_app.txt
python -m pip install pygrabber
python live_capture_app.py
```

A `pygrabber` a DirectShow alapú kameralista Windows-on; nélküle index alapú választás marad.

## Tanítás

```bash
.venv\Scripts\activate
python -m pip install -r requirements_train.txt
# Jupyter: colon_short_szd_r.ipynb (DeepLabV3+), colon_short_szd_r U-net.ipynb (U-Net)
```

A checkpointok alapértelmezés szerint a `checkpoints_0409/` mappába kerülnek; az alkalmazás ezeket tölti be.

## Upstream alap

A szegmentáló architektúra és a tanítási váz a [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) nyílt forráskódú projektre épül (MIT licenc). A saját hozzájárulások: SZE adatintegráció, patient-ID alapú felosztás, élő capture alkalmazás, heterogén PyTorch + TensorFlow pipeline, SQLite naplózás.

## Kapcsolódó fájlok a repóban

- `live_capture_app.py` — fő alkalmazás (élő feed + gombnyomásos elemzés)
- `network/` — DeepLabV3+ implementáció (upstream alap)
- `datasets/custom_colon.py` — SZE adathalmaz betöltő
- `classificator_models/` — ResNet50V2 klasszifikátor (`.keras`, nem a repóban)
- `szakdolgozat/` — szakdolgozat LaTeX forrás és PDF
