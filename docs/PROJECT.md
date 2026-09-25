# PROJECT — kanonikus projektbiblia

Ez a fájl a [birkaslevente/colon_polyp](https://github.com/birkaslevente/colon_polyp) repó kanonikus leírása. Feature- és tanítási munka előtt ezt kell alapnak venni. A szöveg a `master` `726debf` állapotát írja le (2026-09-25, „A live capture alkalmazás legújabb állapota: ViT út, magyarázat és Windows csomagolás”). Forrás: `live_capture_app.py`, `utils/vit_classifier.py`, `utils/explainability.py`, `APP_USAGE_HU.md`, a deploy-fájlok és a `.gitignore`. Ha a kód és ez a dokumentum eltér, a futásidejű igazság a kód; a bibliát ugyanabban a PR-ban kell hozzáigazítani.

A `726debf` commit üzenete szerint kép, dataset, teszt és súly nincs abban a commitban. A követett fa a kódot és a ViT architektúrát tartalmazza (`utils/vit_classifier.py`). A `vit_final_224.pth` nincs a klónban.

A repó publikus, alapértelmezett ág: `master`. Licenc: MIT (`LICENSE`, Copyright 2020 Gongfan Fang — az upstream DeepLabV3Plus-Pytorch licence). A README szerint a szegmentáló váz a [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) projektre épül.

## 1. Cél, scope, határok

Döntéstámogató eszköz vastagbél-polip képalkotáshoz. A pipeline élő kép, videó vagy képszimuláció bemenetéből osztályoz. ResNet50V2 úton előtte szegmentál és ROI-t vág. ViT-B/16 úton a szegmentálás nem fut.

A szoftver **döntéstámogatás**. Klinikai diagnózist és érzékenység/specificitás-ígéretet a repó nem tartalmaz, és ilyet commitolni sem szabad. Betegazonosítóra alkalmas anyag (PHI) nem kerülhet a publikus fába.

Scope-on kívül, amíg külön, elfogadott feladat nincs rá:

- négyosztályos JNET-döntés az élő appban (mindkét klasszifikátor bináris),
- TensorRT / ONNX / kvantálás / desztilláció bevezetése (cél, nem kész munka),
- multimodális LLM finomhangolás.

## 2. Feladat

Mindkét élő klasszifikátor ugyanazt a két belső címkét használja (`CLASSIFIER_CLASS_NAMES` és `utils/vit_classifier.py` `CLASS_NAMES`):

| Index | Belső név | Jelentés |
|------:|-----------|----------|
| 0 | `Non-neoplastic (JNET 1)` | nem neoplasztikus |
| 1 | `Neoplastic (JNET 2a/2b/3)` | neoplasztikus; a 2A, 2B és 3 egy kalap alá esik |

A GUI ezeket JNET nélkül írja ki (`display_class_name`): `Nem neoplasztikus`, `Neoplasztikus`, üres maszknál `Nincs polip`.

Küszöb mindkét úton 0.5:

- **ViT-B/16:** softmax `p1` (neoplastic). `p1 >= 0.5` → neoplasztikus; a konfidencia neoplasztikusnál `p1`, különben `p0`.
- **ResNet50V2:** egy sigmoid kimenet (`malignant_prob`). `>= 0.5` → neoplasztikus, a konfidencia ez a valószínűség; alatta a konfidencia `1.0 - malignant_prob`.

A tanító notebookok a nyers címkét négy Excel-oszlopból olvassák (`JNET_1`, `JNET_2A`, `JNET_2B`, `JNET_3`, érték `'x'`). Az élő döntés ettől függetlenül bináris. Klasszifikátor-tanító notebook a publikus fában nincs.

## 3. Architektúra

Két klasszifikátor-backend van a `live_capture_app.py`-ban. A `Modell` legördülő vált köztük. Ha a ViT súlyfájl betöltődött, az **alap backend a ViT**.

```
kamera | videó (.mp4 .avi .mov .mkv) | images/ szimuláció
        │  VideoCaptureThread  (Windows: pygrabber / DirectShow, különben index)
        ▼
BGR → RGB → ExtResize(513) → ExtCenterCrop(513) → ImageNet normalizálás
        │
        ├─ Modell: ViT-B/16     (alap, ha a .pth megvan)
        │     szegmentálás nem fut (Seg_Model a naplóban: none)
        │     a 513-as kép → Resize(224)  — nem white-bg ROI
        │     softmax p1, küszöb 0.5
        │
        └─ Modell: ResNet50V2
              Szegmentáló: DeepLabV3+ vagy U-Net
              fehér hátterű bbox ROI → 512 (vagy metadata) → Keras sigmoid, küszöb 0.5
        ▼
Magyarázat: ViT attention | ResNet Grad-CAM | szegmentálás Grad-CAM
        ▼
UI + performance_log.csv + saved_results/ + predictions.db
```

Belépő: `python live_capture_app.py` (exe alatt `live_capture_bootstrap.py`). Ablakcím: `HDMI Live Polyp Segmentation`. Elemzés: `Elemzés (Space)` vagy `Space`. Részletes gombok: `APP_USAGE_HU.md`.

PyTorch eszköz: `cuda`, ha elérhető, különben `cpu`. A Keras betöltés a TensorFlow GPU-t elrejteni próbálja (`tf.config.set_visible_devices([], "GPU")`).

## 4. Modellek és checkpoint-elvárások

### ViT-B/16 (PyTorch) — élő klasszifikátor

Forrás: `utils/vit_classifier.py`, `build_vit_b16` / `load_vit_checkpoint`.

| | |
|--|--|
| Architektúra | `torchvision.models.vit_b_16(weights=None, image_size=224)`, a fej `Linear → 2` osztály |
| Súly | state_dict-only `.pth` |
| Útvonal | `VIT_MODEL_PATH` (ha létező fájl), különben `classificator_models/vit_gui_share/vit_final_224.pth` |
| Bemenet | 224×224, ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Transzformáció | `Resize((224, 224))`, utána `ToTensor` + `Normalize`. A ViT modulban nincs CenterCrop |
| Kép | full-frame NBI a modul szerint: a már 513×513-ra vágott megjelenítési kép egésze. Fehér hátterű ROI-t a ViT út nem használ |
| Döntés | softmax `p1`, küszöb `THRESHOLD = 0.5` |
| Név a naplóban | `active_cls_model_name()` → `ViT-B/16` |

Ha a `.pth` megvan és betöltődik, `classifier_backend = "vit"`, és a szegmentáló forward **nem fut**. A `Szegmentáló` menü ilyenkor le van tiltva. Hiányzó vagy hibás ViT-fájl mellett a backend Kerasra esik, ha az a `.keras` betöltődött.

A `vit_final_224.pth` a klónban nincs. A követett kód az architektúrát tartalmazza, a súlyt nem.

### ResNet50V2 (Keras `.keras`) — élő klasszifikátor, csak ROI-úton

Betöltés: `tf.keras.models.load_model`, `compile=False`. Custom object: `ResNetPreprocessLayer` (`package="PolpyCLI"`) és `resnet_v2.preprocess_input`.

Útvonal (`_resolve_classifier_keras_path`):

1. `CLASSIFIER_MODEL_PATH`, ha létező fájl,
2. `classificator_models/cnn_s2_lr2e4_tb_os.keras`,
3. legacy konstans: `classificator_models/resnet50v2_polyp_20260217_131050.keras`,
4. a mappa legfrissebb `*.keras` fájlja.

Alap bemenet: 512×512. Négyzetes `input_shape` a `<stem>_metadata.json`-ból felülírja. A `*_history.json` a futó appot nem érinti.

Ezen az úton fut a szegmentálás. A ROI fehér hátterű (255), a maszk bbox-a, `cv2.resize` (`INTER_LINEAR`).

Hiányzó `.keras` vagy TensorFlow import hiba a teljes appot nem állítja le: a Keras klasszifikátor `None` marad. Ha egyik klasszifikátor sem töltődik, a státusz: „Klasszifikátor: nincs betöltve”.

### Szegmentáció (PyTorch) — csak a ResNet / Keras úton

| Szerep | Konstruktor | Checkpoint |
|--------|-------------|------------|
| DeepLabV3+ | `network.modeling.deeplabv3plus_mobilenet(num_classes=2, output_stride=16)` | `checkpoints_0409/best_model_0409_epoch_31.pth` |
| U-Net | `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)` | `checkpoints_unet/best_model_epoch_22.pth` |

Közös szegmentáló bemenet: `INPUT_SIZE = 513`, ugyanaz az ImageNet normalizálás. A U-Net tanító notebook ImageNet encoder-súlyokkal inicializál; az app `encoder_weights=None`, majd checkpointot tölt. Hiányzó checkpointnál a `_load_checkpoint` figyelmeztet, és a modellt súlyok nélkül adja vissza.

ViT módban ezek a modellek be vannak töltve (warm-up lefut), a `MedicalAnalyzer._analyze_core` viszont nem hívja őket.

### Mi van a klónban a súlyokból

A követett fában nincs `.pth` és nincs `.keras`. A `.gitignore` a `*.pth`, a `*.keras` és a `classificator_models/` útvonalat kizárja. Két szegmentáló checkpointnevet kivételként megenged (`best_model_0409_epoch_31.pth`, `best_model_epoch_22.pth`); ezek sincsenek követve. A ViT súly sem kivétel, sem követett fájl.

A `deploy_exe/MODELLOK_MASOLASA.txt` a DeepLab, a U-Net és a Keras `.keras` másolását írja, valamint a `CLASSIFIER_MODEL_PATH` változót. A ViT útvonal (`vit_gui_share/vit_final_224.pth`, `VIT_MODEL_PATH`) ebben a fájlban nincs. A ViT elhelyezése a kódban és az `APP_USAGE_HU.md`-ben van; a MODELLOK fájl a ViT sorra elavult lehet.

## 5. Inferencia

Belépő: `MedicalAnalyzer.analyze` → `_analyze_core`.

Közös előfeldolgozás mindkét backend előtt: BGR→RGB, `ExtResize(513)`, `ExtCenterCrop(513)`, `ExtToTensor`, `ExtNormalize`. Ebből készül a 513×513-as `orig_img_np`.

### ViT mód

1. Szegmentáló forward nincs. A megjelenített kép a teljes 513-as frame. `seg_conf = 0`.
2. `preprocess_rgb_uint8(orig_img_np)` → `predict_vit`.
3. `performance_log.csv`: `Seg_Model = none`, `Cls_Model = active_cls_model_name()` (`ViT-B/16`), a szegmentálás ideje 0.

### ResNet / Keras mód

1. U-Net: `sigmoid > 0.5`. DeepLab: `argmax` (`output.max(1)[1]`; két osztálynál a polip az 1-es index). A szegmentációs konfidencia a class-1 softmax (U-Netnél a sigmoid) átlaga a maszkon.
2. Fekete hátterű megjelenítés. A klasszifikátornak külön fehér hátterű kép, bbox ROI, resize a Keras bemenetére.
3. `_classify_roi`: sigmoid, küszöb 0.5. Üres maszk: `No Polyp Detected` (a GUI-n `Nincs polip`). Keras objektum `None`: `Classifier unavailable`.
4. Napló: `Seg_Model` = `deeplab` vagy `unet`, `Cls_Model` = `active_cls_model_name()` (`ResNet50V2`, vagy `none`, ha Keras sincs).

CSV oszlopok: `Timestamp`, `Seg_Model`, `Cls_Model`, `Total_Time_ms`, `Preprocess_ms`, `Segmentation_Inference_ms`, `Postprocess_ms`, `Classification_ms`.

SQLite (`saved_results/predictions.db`, tábla `predictions`, WAL): a `class_name` a modell belső címkéje. A `cls_model` oszlop a `_save_results`-ban jelenleg a rögzített `ResNet50V2` szöveg, nem az `active_cls_model_name()`. A CSV és a SQLite felirata ezért eltérhet ViT mentésnél.

### Magyarázat

`utils/explainability.py`, hívás: `MedicalAnalyzer.compute_heatmap`.

| UI (`Magyarázat`) | Mód | Mit rajzol |
|-------------------|-----|------------|
| Klasszifikáció | `classification` | ViT: az utolsó encoder blokk CLS→patch attentionja (`vit_attention_map`), 224-ről 513-ra nagyítva. ResNet: Grad-CAM a Keras konvolúción, a ROI-ból a teljes frame-re képezve |
| Szegmentálás | `segmentation` | PyTorch Grad-CAM a szegmentálón (`grad_cam_torch`). ViT úton a context `input_tensor`-ja `None`, így ez a heatmap nem készül |
| Ki | `off` | overlay nincs |

Az `M` billentyű a kiválasztott magyarázat és a `Ki` között vált. A `Szegmentálás` menüpont ViT backend mellett a forwardot nem pótolja.

## 6. Adat, labelformátum, ami nincs a fában

A notebookok (`colon_short_szd_r.ipynb` és a U-Net párja) helyi útvonalakat várnak:

- címke-Excel: `images/PI-program-JNET-classes.xlsx`,
- képek az Excel mappájában, maszkok: `images/masks`,
- felosztás: fix páciens-ID listák; egy ID egy halmazba kerül,
- a notebook tartalmaz `KvasirSEGDataset` osztályt; a `Kvasir-SEG/` ignorálva van.

Ezek nincsenek a publikus fában. Képszámot, páciensszámot, IoU-t, accuracy-t és latenciát ez a biblia nem rögzít.

A `datasets/` csomag csak az upstream VOC- és Cityscapes-betöltőt exportálja. A `datasets/custom_colon.py` a README-ben szerepel, a fában nincs.

Ignorált vagy hiányzó:

- klinikai képek és maszkok,
- `*.pth`, `*.keras`, `*.h5`, `*.onnx`, `*.pt`, benne a `vit_final_224.pth` és a Keras súlyok,
- `classificator_models/`, `szakdolgozat/`, `tdk/`,
- futásnaplók: `performance_log.csv`, `saved_results/`, `LiveCapture_crash.log`.

PHI és klinikai kép nem kerülhet commitba.

Követett videófájl a gyökérben: `videoplayback (1).mp4`. Tartalmát ez a dokumentum nem minősíti.

## 7. Környezet

Javasolt Python: **3.11.x**. A `deploy_exe/README_HU.md` a helyi `zenbook_venv`-et 3.11.5-ként írja le; a venv nincs a repóban.

| Fájl | Mire |
|------|------|
| `requirements_app.txt` | GUI: numpy, pillow, opencv-python, matplotlib, customtkinter, torch, torchvision, segmentation-models-pytorch, tensorflow |
| `requirements_train.txt` | A két szegmentáló notebook. TensorFlow nincs ebben a listában |
| `deploy_exe/requirements_build.txt` | app requirements + `pyinstaller>=6.0` |
| `pygrabber` | nincs a requirements fájlban; Windows DirectShow kameralistához külön telepítés |

### UI

CustomTkinter. `choose_fixed_window`: a monitor aránya 16:9, 16:10 vagy 4:3, az ablak ehhez igazodik, utána `resizable(False, False)`. A 16:10 elrendezése a 16:9-ével egyezik. Magyar vezérlők: `Élő indítás`, `Szimuláció`, `Elemzés (Space)`, `Szegmentáló`, `Modell`, `Magyarázat`. A `Szegmentáló` csak ResNet backendnél aktív. A `Modell` lista csak a ténylegesen betöltött klasszifikátorokat mutatja.

### Deploy

Windows-fókusz.

- `scripts/install_dev.ps1` — dev venv (`zenbook_venv`), PyTorch alapból CPU wheel, `-GpuTorch` CUDA 12.4, `-SkipTensorFlow`.
- `live_capture_bootstrap.py` — PyInstaller belépő, crash-log az exe mellett és a `%TEMP%` alatt. Követett fájl.
- `deploy_exe/build_onedir.ps1`, `deploy_exe/live_capture_app.spec` — onedir build.
- `deploy_exe/build_installer.ps1`, `installer/LiveCapture.iss` — Inno Setup varázsló (`LiveCapture-setup`), ha az `ISCC.exe` elérhető. Részletek: `deploy_exe/README_HU.md`.

A `deploy_exe/README_HU.md` még említi az `install.bat`, `build_exe.bat` és `build_exe_console.bat` fájlokat. A `**/*.bat` ignorálva van, ezek a fájlok nincsenek a követett fában. A `requirements_app.txt` a `scripts/tools/visualize_features.py` fájlt említi; az sincs a fában.

## 8. Mérési és A/B szabályok

A repóban nincs rögzített, idézhető accuracy / sensitivity / specificity baseline. Gyorsítási munka előtt a viszonyítási alap a helyi DeepLab-, U-Net-, Keras- és ViT-súly, ugyanazon a kiértékelő halmazon, backendenként külön.

Szabály, amíg Levente mást nem mond:

1. Először baseline mérés (accuracy, sensitivity, specificity; szegmentációnál a notebookban használt IoU), utána A/B ugyanazzal a protokollal.
2. Laptopos gyorsítás (diszkrét vagy integrált GPU) nem járhat pontosságromlással Levente tudta nélkül.
3. Előnyben részesített irány, ha publikus git és cikk alátámasztja: TensorRT, ONNX, kvantálás, desztilláció, könnyebb architektúra. Ezek még nincsenek a futásidejű útvonalon.
4. A `performance_log.csv` latenciát mér, pontosságot nem. ViT sorban a szegmentálás ideje 0 és a `Seg_Model` `none`. A Keras GPU-tiltás sikertelensége esetén a klasszifikátor eszköze eltérhet; a mérésnél rögzíteni kell.

Számokat csak mérés után, a mérés leírásával együtt szabad a bibliába vagy a PR-be írni.

## 9. Workflow

- Új munka: feature branch + pull request. A `master` csak Levente kifejezett igenjével változik.
- Egy PR egy feladathoz. A biblia és a kód együtt mozog, ha a viselkedés változik.
- Klinikai állítás és PHI commit tilos.
- Kapcsolódó leírások: `README.md`, `APP_USAGE_HU.md` (gombok, ViT vs ResNet), `REPRODUCIBILITY.md`, `deploy_exe/README_HU.md`. A súlymásolás ViT sora az `APP_USAGE_HU.md`-ben és a kódban van; a `MODELLOK_MASOLASA.txt` ettől lemaradhat.

Következő termékcélok (nincs kész implementáció a fában):

- gyorsabb laptop-inferencia pontosságromlás nélkül,
- később UI-polírozás a jelenlegi fix ablakon túl,
- később multimodális LLM finomhangolási út.

## 10. Nyitott rések

| Hivatkozás | Hol szerepel | A fában |
|------------|--------------|---------|
| `datasets/custom_colon.py` | `README.md` | nincs; `datasets/__init__.py` csak VOC + Cityscapes |
| `classificator_models/` és `vit_final_224.pth` | app, `APP_USAGE_HU.md` | gitignore; az architektúra (`utils/vit_classifier.py`) követett, a súly nem |
| Keras `.keras` súlyok | app, `MODELLOK_MASOLASA.txt` | gitignore, nincs követett fájl |
| `szakdolgozat/` | `README.md` | gitignore, nincs a fában |
| klasszifikátor-tanító notebook | — | a publikus notebooklista csak a két szegmentáló notebook |
| `install.bat`, `build_exe.bat`, `build_exe_console.bat` | `deploy_exe/README_HU.md` | `**/*.bat` ignorálva, nincs a fában |
| `scripts/tools/visualize_features.py` | `requirements_app.txt` fejléc | nincs a fában |
| `scripts/README.md` | `README.md` táblázat | nincs a fában |
| a két best szegmentáló checkpoint | app + gitignore-kivétel | nincs követve |
| ViT sor a `MODELLOK_MASOLASA.txt`-ben | — | a fájl a Keras/szegmentáló súlyokat sorolja; a ViT path a kódban és az `APP_USAGE_HU.md`-ben van |
| SQLite `cls_model` | `App._save_results` | a CSV `active_cls_model_name()`-et ír; a DB insert a `ResNet50V2` literált írja |

A DeepLab-notebook egyik kiértékelő cellája `checkpoints_0409/best_model_0409_epoch_38.pth` hiányát naplózza. Az app az epoch 31-es checkpointot tölti. A két fájlnév nincs közös, dokumentált kiválasztási szabállyal összekötve.
