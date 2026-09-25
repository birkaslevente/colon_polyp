# PROJECT — kanonikus projektbiblia

Ez a fájl a [birkaslevente/colon_polyp](https://github.com/birkaslevente/colon_polyp) repó kanonikus leírása. Feature- és tanítási munka előtt ezt kell alapnak venni. A szöveg a `master` fa 2026-09-25-i állapotát írja le (`live_capture_app.py`, notebookok, `README.md`, `REPRODUCIBILITY.md`, `APP_USAGE_HU.md`, `deploy_exe/`). Ha a kód és ez a dokumentum eltér, a futásidejű igazság a kód; a bibliát ugyanabban a PR-ban kell hozzáigazítani.

A repó publikus, alapértelmezett ág: `master`. Licenc: MIT (`LICENSE`, Copyright 2020 Gongfan Fang — az upstream DeepLabV3Plus-Pytorch licence). A README szerint a szegmentáló váz a [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) projektre épül.

## 1. Cél, scope, határok

Döntéstámogató eszköz vastagbél-polip képalkotáshoz. A pipeline élő kép, videó vagy képszimuláció bemenetéből szegmentál, ROI-t vág, majd bináris JNET-osztályt ad.

A szoftver **döntéstámogatás**. Klinikai diagnózist, orvosi eszköz-állítást és érzékenység/specificitás-ígéretet a repó nem tartalmaz, és ilyet commitolni sem szabad. Betegazonosítóra alkalmas anyag (PHI) nem kerülhet a publikus fába.

Scope-on kívül, amíg külön, elfogadott feladat nincs rá:

- négyosztályos JNET-döntés az élő appban (a klasszifikátor bináris),
- TensorRT / ONNX / kvantálás / desztilláció bevezetése (cél, nem kész munka),
- multimodális LLM finomhangolás,
- UI-polírozás a jelenlegi CustomTkinter felületen túl.

## 2. Feladat

A klasszifikátor két címkét ismer. Forrás: `CLASSIFIER_CLASS_NAMES` a `live_capture_app.py`-ban.

| Index | Megjelenített név | Jelentés a kódban |
|------:|-------------------|-------------------|
| 0 | `Non-neoplastic (JNET 1)` | nem neoplasztikus |
| 1 | `Neoplastic (JNET 2a/2b/3)` | neoplasztikus; a 2A, 2B és 3 egy kalap alá esik |

A Keras-modell egyetlen sigmoid kimenetet ad. A döntés a `MedicalAnalyzer.classify` függvényben: `malignant_prob >= 0.5` → neoplasztikus, és a visszaadott konfidencia ez a valószínűség. Alatta a címke nem neoplasztikus, a konfidencia `1.0 - malignant_prob`. Ha a klasszifikátor objektum `None`, a szöveg `Classifier unavailable`. Ha a ROI `None`, a függvény a nem neoplasztikus címkét adja 0.0 konfidenciával. Ha a maszk üres, az `analyze` a `No Polyp Detected` szöveget adja, klasszifikáció nélkül.

A tanító notebookok a nyers címkét négy Excel-oszlopból olvassák (`JNET_1`, `JNET_2A`, `JNET_2B`, `JNET_3`, érték `'x'`). Ez a nyers címkeforma. Az élő app döntése ettől függetlenül bináris. Az élő slotba ResNet50V2 és helyi ViT `.keras` is tehető (4. fejezet).

## 3. Architektúra

```
kamera | videó (.mp4 .avi .mov .mkv) | images/ szimuláció
        │  VideoCaptureThread  (Windows: pygrabber / DirectShow, különben index)
        ▼
MedicalAnalyzer.analyze(frame_bgr)
        │
        ├─ BGR → RGB → ExtResize(513) → ExtCenterCrop(513)
        │    → ExtToTensor → ImageNet normalizálás
        ▼
szegmentáció (futás közben váltható)
        ├─ DeepLabV3+ MobileNet   PyTorch, CUDA ha van, különben CPU
        └─ U-Net ResNet34         ugyanígy
        ▼
fehér hátterű maszkolt polip → bbox ROI → resize a klasszifikátor bemenetére
        ▼
egy Keras .keras slot (bináris JNET) — ResNet50V2 vagy helyi ViT .keras
        a betöltés a TF GPU-t elrejteni próbálja
        ▼
UI (CustomTkinter) + performance_log.csv + saved_results/ + predictions.db
```

Belépő: `python live_capture_app.py`, osztály `App`, ablakcím `HDMI Live Polyp Segmentation`. Elemzés: `Capture & Analyze` vagy `Space`. Modellváltás: `Model` gomb → `MedicalAnalyzer.switch_model` (`deeplab` | `unet`). Alapértelmezett szegmentáló: DeepLab.

PyTorch eszköz: `cuda`, ha elérhető, különben `cpu`. A klasszifikátor betöltésekor a kód `tf.config.set_visible_devices([], "GPU")` hívással a TensorFlow GPU-t elrejti.

## 4. Modellek és checkpoint-elvárások

### Szegmentáció (PyTorch)

| Szerep | Konstruktor | Checkpoint (az app mappája mellett) |
|--------|-------------|-------------------------------------|
| DeepLabV3+ | `network.modeling.deeplabv3plus_mobilenet(num_classes=2, output_stride=16)` | `checkpoints_0409/best_model_0409_epoch_31.pth` (`DEEPLAB_CHECKPOINT`) |
| U-Net | `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)` | `checkpoints_unet/best_model_epoch_22.pth` (`UNET_CHECKPOINT`) |

Közös bemenet: `INPUT_SIZE = 513`. Normalizálás: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

A `_load_checkpoint` a `model_state_dict`, `model_state` vagy `state_dict` kulcsot, ezek hiányában a nyers state dictet tölti. Hiányzó fájlnál figyelmeztet, és a modellt súlyok nélkül adja vissza (nem állítja le a folyamatot ezen a ponton).

A U-Net **tanító** notebook ImageNet encoder-súlyokkal inicializál (`encoder_weights="imagenet"`). Az **app** `encoder_weights=None`-nal épít, majd a checkpointot tölti.

### Klasszifikátor (TensorFlow / Keras)

Az élő úton egyetlen klasszifikátor-slot van. A slot egy bináris Keras `.keras` fájl: JNET 1 (nem neoplasztikus) szemben a JNET 2a/2b/3 összevont neoplasztikus címkével. A döntés továbbra is egy sigmoid kimenet, küszöb 0.5 (2. fejezet).

Levente (CEO) szerint ebbe a slotba a **ResNet50V2 és a ViT is** érvényes élő opció, ha a megfelelő helyi `.keras` a helyén van. A szegmentáló (DeepLab vagy U-Net) és a klasszifikátor (ResNet50V2 vagy ViT) párosai egyaránt használhatók; a párok viselkedése közel azonos. Ehhez a mondathoz a repóban nincs számszerű összehasonlítás. A `Model` gomb csak a szegmentálót cseréli futás közben. A klasszifikátor a betöltéskor választott egyetlen `.keras`.

A követett `live_capture_app.py` ViT-osztályt, ViT-importot és külön architektúra-ágat nem tartalmaz. A `tf.keras.models.load_model` azt a fájlt tölti, amit az útvonal-feloldás ad (`compile=False`). A regisztrált custom object a `ResNetPreprocessLayer` (`package="PolpyCLI"`) és a `resnet_v2.preprocess_input`. A fájlfejléc kommentje, a `performance_log.csv` `Cls_Model` oszlopa és a SQLite `cls_model` mezője a `ResNet50V2` szöveget írja akkor is, ha a betöltött fájl más architektúra. Ez felirat, nem architektúra-detektálás.

ViT használata helyi súlycsere: a `.keras` a `CLASSIFIER_MODEL_PATH` vagy a `classificator_models/` alatt, plusz a bemeneti méret (`*_metadata.json`, különben 512) és a preprocess illesztése a mentett modellhez. A publikus fa ViT-súlyt nem tartalmaz.

Útvonal-feloldás (`_resolve_classifier_keras_path`), ebben a sorrendben:

1. `CLASSIFIER_MODEL_PATH`, ha létező fájl,
2. preferált név: `classificator_models/cnn_s2_lr2e4_tb_os.keras`,
3. legacy konstans, a `resnet50v2_polyp_*.keras` család egy konkrét fájlja: `classificator_models/resnet50v2_polyp_20260217_131050.keras`,
4. ha az előző kettő nincs a mappában: a `classificator_models/` legfrissebb `*.keras` fájlja.

Alap bemenet: 512×512 (`CLASSIFIER_INPUT_SIZE`). Ha a modell mellett van `<stem>_metadata.json` és az `input_shape` négyzetes, az oldalél onnan jön. A `*_history.json` a futó appot nem érinti.

TensorFlow **import** hiba esetén a klasszifikátor `None` marad, a szegmentálók betöltődhetnek, az app elindul. Ha a TensorFlow import sikeres, de a `.keras` hiányzik vagy a `load_model` hibázik, a `load_models` egészében sikertelen (`models_loaded = False`).

### Mi van a klónban a súlyokból

A `git ls-files` kimenetében nincs `.pth` és nincs `.keras`. A `.gitignore` két konkrét checkpointnevet kivételként megenged (`checkpoints_0409/best_model_0409_epoch_31.pth`, `checkpoints_unet/best_model_epoch_22.pth`), ezek a jelenlegi fában nincsenek követve. A `classificator_models/` mappa ignorálva van. A súlyok elhelyezése: `deploy_exe/MODELLOK_MASOLASA.txt`. PyInstaller alatt a `_BASE_DIR` az `.exe` könyvtára.

## 5. Inferencia

Függvény: `MedicalAnalyzer.analyze` (`live_capture_app.py`).

1. **Előfeldolgozás.** BGR→RGB, PIL, `preprocess`: `ExtCompose` (`ExtResize` 513, `ExtCenterCrop` 513, `ExtToTensor`, `ExtNormalize`). Dummy maszkot kap, mert az `ext_transforms` `(img, mask)` párt vár.
2. **Szegmentáció** (`torch.no_grad`).
   - U-Net: `sigmoid`, maszk ahol a valószínűség `> 0.5`.
   - DeepLab: maszk = `argmax` az osztálytengelyen (`output.max(1)[1]`; két osztálynál a polip az 1-es index). A szegmentációs konfidencia a class-1 softmax átlaga a maszk pixelein.
3. **Utófeldolgozás.** A tensor denormalizálása uint8 képpé. A megjelenített maszkolt kép háttere fekete. A klasszifikátornak külön, 255-ös fehér hátterű kép készül, rajta a polip pixelei. `_extract_roi`: a maszk bbox-a, `cv2.resize` (`INTER_LINEAR`) a klasszifikátor oldalhosszára, `float32`.
4. **Klasszifikáció.** `classify`: batch dimenzió, `predict`, egy sigmoid skalár, küszöb 0.5 (lásd 2. fejezet). A betöltött `.keras` lehet ResNet50V2 vagy helyi ViT; a hívás ugyanaz.
5. **Napló.** `performance_log.csv` oszlopok: `Timestamp`, `Seg_Model`, `Cls_Model` (a kód mindig a `ResNet50V2` szöveget írja, ViT fájlnál is), `Total_Time_ms`, `Preprocess_ms`, `Segmentation_Inference_ms`, `Postprocess_ms`, `Classification_ms`.
6. **Mentés** (`App._save_results`). Képek a `saved_results/` alá. SQLite: `saved_results/predictions.db`, tábla `predictions` (`timestamp`, `class_name`, `confidence`, `seg_confidence`, `seg_model`, `cls_model`, `input_path`, `mask_path`, `mask_blob`), WAL mód.

## 6. Adat, labelformátum, ami nincs a fában

A notebookok (`colon_short_szd_r.ipynb` és a U-Net párja) helyi útvonalakat várnak:

- címke-Excel: `images/PI-program-JNET-classes.xlsx`,
- képek: az Excel mappája, maszkok: `images/masks`,
- felosztás: fix páciens-ID listák (`train_ids` / `val_ids` / `test_ids`); egy ID egy halmazba kerül,
- a notebook tartalmaz `KvasirSEGDataset` osztályt is; a `Kvasir-SEG/` a `.gitignore`-ban van.

Ezek a fájlok és mappák nincsenek a publikus fában. Képszámot, páciensszámot és IoU-t ez a biblia nem rögzít: a notebook-kimenetek egy adott futás nyomai, nem publikált baseline.

A `datasets/` csomag csak az upstream VOC- és Cityscapes-betöltőt exportálja (`datasets/__init__.py`). A `datasets/custom_colon.py` a README-ben szerepel, a fában nincs, és a csomag nem importálja.

Ignorált vagy hiányzó, amit commitolni tilos, illetve ami jelenleg nincs a tree-ben:

- klinikai képek és maszkok (`images/`, `Kvasir-SEG/`, és a `.gitignore` egyéb adatútvonalai),
- `*.pth`, `*.keras`, `*.h5`, `*.onnx`, `*.pt` (a fenti két checkpoint-kivétel nincs követve),
- `classificator_models/`, `szakdolgozat/`, `tdk/`,
- futásnaplók: `performance_log.csv`, `saved_results/`, `LiveCapture_crash.log`.

PHI és klinikai kép nem kerülhet commitba.

A fában követett, a fenti listán kívüli nagyobb fájl: `videoplayback (1).mp4` a gyökérben. Tartalmát ez a dokumentum nem minősíti.

## 7. Környezet

Javasolt Python: **3.11.x**. A `deploy_exe/README_HU.md` a helyi `zenbook_venv`-et 3.11.5-ként írja le; ez a venv nincs a repóban (gitignore).

| Fájl | Mire |
|------|------|
| `requirements_app.txt` | GUI app: numpy, pillow, opencv-python, matplotlib, customtkinter, torch, torchvision, segmentation-models-pytorch, tensorflow |
| `requirements_train.txt` | A két notebook: a fentiek tanításhoz (pandas, tqdm, scikit-learn, openpyxl, visdom, jsonschema, …). TensorFlow nincs ebben a listában |
| `deploy_exe/requirements_build.txt` | `requirements_app.txt` + `pyinstaller>=6.0` |
| `pygrabber` | nincs a requirements fájlban; Windows DirectShow kameralistához külön `pip install` |

Windows-fókusz: `pygrabber`, `scripts/install_dev.ps1` (venv neve alapból `zenbook_venv`, PyTorch alapból CPU wheel, `-GpuTorch` CUDA 12.4, `-SkipTensorFlow` opció), `deploy_exe/build_onedir.ps1` és `deploy_exe/live_capture_app.spec` (PyInstaller **onedir**). A spec belépője `live_capture_bootstrap.py`. A `deploy_exe/README_HU.md` említi az `install.bat`, `build_exe.bat` és `build_exe_console.bat` fájlokat is; a `**/*.bat` ignorálva van, és ezek a fájlok, valamint a `live_capture_bootstrap.py` és a requirementsben említett `visualize_features.py` nincsenek a követett fában.

## 8. Mérési és A/B szabályok

A repóban nincs rögzített, idézhető accuracy / sensitivity / specificity baseline. Gyorsítási munka előtt a mai DeepLab-, U-Net- és Keras-checkpoint a viszonyítási alap, ugyanazon a kiértékelő halmazon.

Szabály, amíg Levente mást nem mond:

1. Először baseline mérés (accuracy, sensitivity, specificity, és a szegmentációnál a notebookban már használt IoU), utána A/B ugyanazzal a protokollal.
2. Laptopos gyorsítás (diszkrét vagy integrált GPU) nem járhat pontosságromlással Levente tudta nélkül. A sebesség nyeresége önmagában nem indok a regresszióra.
3. Előnyben részesített irány, ha publikus git és cikk alátámasztja: TensorRT, ONNX, kvantálás, desztilláció, könnyebb architektúra. Ezek még nincsenek a futásidejű útvonalon.
4. A `performance_log.csv` latenciát mér (total / pre / seg / post / cls), pontosságot nem. A betöltés a TensorFlow GPU-t elrejteni próbálja; ha a `set_visible_devices` kivételt dob, a kód csendben továbbmegy. A mérésnél a klasszifikátor eszközét külön kell rögzíteni.

Számokat csak mérés után, a mérés leírásával együtt szabad a bibliába vagy a PR-be írni.

## 9. Workflow

- Új munka: feature branch + pull request. A `master` csak Levente kifejezett igenjével változik.
- Egy PR egy feladathoz. A biblia és a kód együtt mozog, ha a viselkedés változik.
- Klinikai állítás és PHI commit tilos.
- Kapcsolódó, már létező leírások: `README.md` (rövid áttekintés), `APP_USAGE_HU.md` (gombok), `REPRODUCIBILITY.md` (környezet), `deploy_exe/README_HU.md` (exe). Ütközés esetén ezt a fájlt kell igazítani, vagy a régebbi dokumentumot ugyanabban a PR-ban.

Következő termékcélok (nincs kész implementáció a fában):

- gyorsabb laptop-inferencia pontosságromlás nélkül,
- később UI-polírozás,
- később multimodális LLM finomhangolási út.

## 10. Nyitott rések

A README vagy a deploy-dokumentáció említi, a követett fa nem tartalmazza:

| Hivatkozás | Hol szerepel | A fában |
|------------|--------------|---------|
| `datasets/custom_colon.py` | `README.md` | nincs; `datasets/__init__.py` csak VOC + Cityscapes |
| `classificator_models/` | README, app, `MODELLOK_MASOLASA.txt` | gitignore, nincs követett fájl |
| `szakdolgozat/` | `README.md` | gitignore, nincs a fában |
| klasszifikátor-tanító notebook | — | a publikus notebooklista csak a két szegmentáló notebook (`colon_short_szd_r.ipynb`, `colon_short_szd_r U-net.ipynb`) |
| `live_capture_bootstrap.py` | `deploy_exe/live_capture_app.spec`, deploy README | nincs követve |
| `install.bat`, `build_exe.bat`, `build_exe_console.bat` | `deploy_exe/README_HU.md` | `**/*.bat` ignorálva, nincs a fában |
| `visualize_features.py` | `requirements_app.txt` fejléc | nincs a fában |
| a két best checkpoint | app + gitignore-kivétel | nincs követve |
| ViT klasszifikátor-súly | Levente: helyi `.keras` az élő slotban | nincs a követett fában; a kód nem épít ViT-architektúrát |

A DeepLab-notebook egyik kiértékelő cellája `checkpoints_0409/best_model_0409_epoch_38.pth` hiányát naplózza. Az app által betöltött fájlnév az epoch 31-es checkpoint. A két fájlnév nincs közös, dokumentált kiválasztási szabállyal összekötve.
