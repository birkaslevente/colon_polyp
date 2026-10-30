# Polip szegmentációs és klasszifikációs élő alkalmazás (SZE szakdolgozat)

Ez a branch csak a `live_capture_app.py` futtatásához közvetlenül szükséges fájlokat tartalmazza.

## Tartalom

| Fájl / mappa | Szerep |
|--------------|--------|
| `live_capture_app.py` | Fő alkalmazás (élő feed + gombnyomásos elemzés) |
| `live_capture_bootstrap.py` | Indító (hibanaplózás, majd az app betöltése) |
| `network/` | DeepLabV3+ implementáció (upstream alap) |
| `utils/ext_transforms.py` | Szegmentációs előfeldolgozás |
| `utils/explainability.py` | Grad-CAM / attention magyarázatok |
| `utils/vit_classifier.py` | ViT-B/16 klasszifikátor |
| `requirements_app.txt` | Futtatási függőségek |
| `APP_USAGE_HU.md` | Használati útmutató, modellfájlok elhelyezése |

## Fontos korlát

Klinikai képek, maszkok és betanított súlyfájlok adatvédelmi és méretkorlát miatt **nem** részei a repónak. A futtatáshoz a modellfájlokat külön kell elhelyezni (lásd `APP_USAGE_HU.md`).

## Indítás

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements_app.txt
python -m pip install pygrabber
python live_capture_app.py
```

A `pygrabber` a DirectShow alapú kameralista Windows-on; nélküle index alapú választás marad.

## Upstream alap

A szegmentáló architektúra a [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) nyílt forráskódú projektre épül (MIT licenc).
