# TDK roadmap — következő forduló

Döntéstámogató eszköz vastagbél-polip képalkotáshoz. A szoftver nem állít fel diagnózist, és érzékenységet vagy specificitást ez a fájl nem ígér. Terminológia: a [#2](https://github.com/birkaslevente/colon_polyp/pull/2) `docs/PROJECT.md` biblia (draft). Futásidejű stack: `master` `726debf`. HarDNet: a [#3](https://github.com/birkaslevente/colon_polyp/pull/3) notebook draft. Egyik draft sincs merge-ölve. Mért accuracy, Dice, IoU és milliszekundum ebben a fájlban nincs, mert a repó ilyet nem rögzít.

## 1. Jelenlegi stack

### Klasszifikáció

Két élő backend van a `live_capture_app.py`-ban. A `Modell` menü vált köztük. Ha a `vit_final_224.pth` betöltődik, az alap a ViT.

| | ViT-B/16 | ResNet50V2 |
|--|----------|------------|
| Kód | `utils/vit_classifier.py`, `torchvision` `vit_b_16`, fej `Linear → 2` | Keras `.keras`, `tf.keras.models.load_model` |
| Súly | `VIT_MODEL_PATH`, különben `classificator_models/vit_gui_share/vit_final_224.pth` | `CLASSIFIER_MODEL_PATH`, különben a `classificator_models/` alatti `.keras` |
| Bemenet | a 513×513-as frame egésze, `Resize((224, 224))`, fehér hátterű ROI nélkül | fehér hátterű bbox ROI, alapból 512×512 |
| Döntés | softmax `p1 ≥ 0.5` | sigmoid `≥ 0.5` |
| Szegmentálás | nem fut (`Seg_Model = none`) | ezen az úton fut |

A belső címke mindkét úton bináris: `Non-neoplastic (JNET 1)` és `Neoplastic (JNET 2a/2b/3)`. A GUI JNET nélkül írja ki. A négy Excel-oszlopos JNET címke csak a tanító notebookokban van; az élő app nem dönt 2A / 2B / 3 között. A súlyok nincsenek a klónban.

### Szegmentáció

Csak a ResNet úton fut forward. A modellek ViT módban is betöltődhetnek, a `_analyze_core` viszont nem hívja őket.

| Szerep | Modell | Checkpoint |
|--------|--------|------------|
| App alap | DeepLabV3+ MobileNet (`deeplabv3plus_mobilenet`, `output_stride=16`, 2 osztály) | `checkpoints_0409/best_model_0409_epoch_31.pth` |
| Váltó | U-Net, `segmentation_models_pytorch` ResNet34, `encoder_weights=None` az appban | `checkpoints_unet/best_model_epoch_22.pth` |

Közös bemenet: `INPUT_SIZE = 513`. A DeepLab notebook egy cellája `best_model_0409_epoch_38.pth` hiányát is naplózza. Az app az epoch 31-es fájlt tölti. A TDK egy nevet használjon.

HarDNet-MSEG a masteren nincs. A #3 draft viszi: `colon_short_szd_r_hardnet_mseg.ipynb`, vendored kód `network/hardnet_mseg/`, checkpoint minta `checkpoints_hardnet/best_model_hardnet_epoch_{N}.pth`. `RUN_TRAINING` és `RUN_BENCHMARK` a PR-ben alapból ki van kapcsolva. Tanítás abban a futásban nem volt. Az app ezt a modellt még nem tölti.

### Alkalmazás

CustomTkinter, ablakcím `HDMI Live Polyp Segmentation`. Belépő: `python live_capture_app.py`. Exe alatt: `live_capture_bootstrap.py`.

- Forrás: kamera (Windowson pygrabber / DirectShow, különben index), videó (`.mp4`, `.avi`, `.mov`, `.mkv`), `images/` szimuláció.
- Elemzés: `Elemzés (Space)` vagy `Space`.
- Magyarázat (`utils/explainability.py`): ViT CLS→patch attention, ResNet Grad-CAM, szegmentálás Grad-CAM. ViT úton a szegmentálás-hőtérkép nem készül, mert az `input_tensor` üres.
- Napló: `performance_log.csv` (latencia), `saved_results/`, `saved_results/predictions.db`. A CSV `Cls_Model` az `active_cls_model_name()`. A SQLite insert a `ResNet50V2` szöveget írja, ViT mentésnél is.
- Windows csomag: `deploy_exe/build_onedir.ps1`, `deploy_exe/live_capture_app.spec`, `deploy_exe/build_installer.ps1`, `installer/LiveCapture.iss` (Inno Setup, ha van `ISCC.exe`). A `.pth` / `.keras` az exe mellett van, nem a repóban. A `deploy_exe/MODELLOK_MASOLASA.txt` a ViT sort nem tartalmazza; a ViT útvonal a kódban és az `APP_USAGE_HU.md`-ben van.
- PyTorch: `cuda`, ha van, különben `cpu`. A Keras betöltés a TensorFlow GPU-t elrejteni próbálja.

### Határ

Döntéstámogatás. Klinikai állítás és PHI a publikus fába nem kerül. A `*.pth`, `*.keras`, `classificator_models/`, `images/` és a klinikai Excel ignorálva van.

## 2. Mit mire cserélni TDK-ig

Az Effort a megvalósítás terjedelme, naptári ígéret nélkül. Alacsony: jegyzőkönyv vagy vékony mérés a meglévő úton. Közepes: egy már megkezdett notebook végigvitele, vagy egy export. Magas: új tanítóstack vagy olyan SDK, ami nincs a fában.

A nyereségoszlop szándékot ír. Szám csak a biblia 8. pontja szerint kerülhet mellé: először baseline ugyanazon a patient-ID spliten, utána A/B. Szegmentációnál a notebook IoU (HarDNetnál a #3 Dice/IoU cellája). Klasszifikációnál accuracy, sensitivity, specificity. Laptopos gyorsítás pontosságromlással csak Levente tudtával mehet.

| Komponens | Jelenlegi | Jelölt csere | Miért / nyereség | Effort | Kockázat | Licence |
|-----------|-----------|--------------|------------------|--------|----------|---------|
| Szegmentáló, első jelölt | DeepLabV3+ MobileNet, app: `checkpoints_0409/best_model_0409_epoch_31.pth`. Mellette U-Net ResNet34, `checkpoints_unet/best_model_epoch_22.pth`. Forward csak ResNet úton. | HarDNet-MSEG. Kód és notebook a #3 drafton: `colon_short_szd_r_hardnet_mseg.ipynb`, `checkpoints_hardnet/`. `RUN_TRAINING = False`, amíg lokálisan nem kapcsolják. | Ugyanaz a patient-ID split és a DeepLab notebook mintája. Egy jelölt elég egy TDK A/B-hez a két meglévő szegmentáló ellen. A saját spliten a Dice, az IoU és a ms ismeretlen. A cikk nyilvános halmazon mért száma ide nem másolható. | Közepes. A váz draftban megvan. Hátra: lokális adat, `hardnet68.pth` a `HARDNET68_PRETRAINED_PATH`-on, tanítás, benchmark. Az appbekötés külön lépés, a mérés után. | Draft, nincs a masteren. Súly a gitignore-on marad. Hiányzó baseline `.pth` esetén a notebook `[SKIP]`, nem eredmény. | Apache-2.0, a #3 `network/hardnet_mseg/LICENSE` (upstream: [james128333/HarDNet-MSEG](https://github.com/james128333/HarDNet-MSEG)). A HarDNet-68 ImageNet `.pth` nincs a repóban ([PingoLH/Pytorch-HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)); ennek a fájlnak a licence a letöltéskor ellenőrizendő, a #3 `UPSTREAM.md` nem rögzíti. |
| Szegmentáló, második jelölt | Ugyanaz a két checkpoint. | [PraNet-V2](https://github.com/ai4colonoscopy/PraNet-V2) (DSRA, arXiv:2504.10986), bináris polipág. Csak a HarDNet-mérés után, ha az első jelölt a saját spliten nem ad értelmezhető A/B-t. | Nyilvános git és cikk. A fában nincs vendored kód, ezért nem az első csere. | Magas. Új train stack, nincs notebook a repóban. | Két idegen kódbázis a beadandó előtt szétviszi a fókuszt. A V1 fájlok átemelése a V1 feltételét is behozza. | A V2 gyökér `LICENSE` MIT (Copyright 2025 ai4colonoscopy; a raw fájl 2026-09-27-én ezt mutatta). A V1 repó ([DengPingFan/PraNet](https://github.com/DengPingFan/PraNet)) szövege: kutatás és oktatás, kereskedelmi használathoz külön engedély. Termék előtt a V2 fát diffelni kell a V1-gyel. TDK-kutatásra hivatkozható; kereskedelmi útra a szigorúbb szöveg a mérvadó, amíg a diff tiszta nem. |
| Klasszifikátor gyorsítás, ONNX | ViT-B/16 PyTorch eager, `utils/vit_classifier.py`, 224×224. | Saját export ONNX-ra, futtatás ONNX Runtime-mal. Architektúra és a privát `.pth` marad. | A biblia 9. pontja ezt a irányt nevezi meg (TensorRT, ONNX, kvantálás), kész kód nélkül. A ms nyereség ismeretlen, amíg ugyanazon a laptopon nincs A/B a `performance_log.csv` `Classification_ms` oszlopa ellen. | Közepes. Fix 224-es bemenet. Windows CPU és CUDA execution provider külön mérés. | A Keras ResNet50V2 másik gráf. Egy ONNX a ViT utat fedi. A ResNet export külön feladat, és a ROI előfeldolgozás a gráfon kívül marad. | ONNX Runtime: MIT. A torchvision ViT kód a PyTorch BSD-3 licence. A `.pth` továbbra sem publikus. |
| Klasszifikátor gyorsítás, TensorRT | Ugyanaz a ViT eager út. GPU csak ha a gépen van CUDA; az `install_dev.ps1` alapból CPU wheel. | TensorRT motor az ONNX-ból, NVIDIA GPU-s laptopon. | Akkor érdemes, ha az ONNX Runtime A/B után a keret még kívül van. Addig a nyereség ismeretlen. | Magas. SDK, driver, a PyInstaller spec. | CPU-only gépen nincs út. Az FP16 pontosság ismeretlen. A motor nem hordozható másik GPU-generációra ellenőrzés nélkül. | NVIDIA saját SDK-licenc, nem OSI. A repó MIT szövege ezt nem fedi. Egyetemi gépen a telepített SDK feltétele számít. |
| Kvantálás | FP32 PyTorch (és Keras a ResNet úton). | FP16 vagy INT8 az ONNX Runtime-on. TensorRT INT8 csak az előző sor után. | Kisebb motor, CPU-n esetleg gyorsabb út. A pontosságdeltát ugyanazzal a protokollal kell mérni. | Közepes, ha az ONNX export megvan. INT8-hoz kalibrációs halmaz kell. | A kalibráló klinikai kép PHI. A halmaz a gépen marad, a repóba nem kerül. A pontosságromlás mértéke ismeretlen. | A választott runtime licence (ORT MIT, TRT NVIDIA). |
| Könnyebb klasszifikátor | ViT-B/16 marad az alap, full-frame 224. | Nincs drop-in a fában. Csak akkor, ha az ONNX ViT a laptop kereten kívül marad: új bináris fej, ugyanazon a két címkén, saját tanítással. Idegen orvosi súly letöltése nem jelölt. | Kisebb modell a gombnyomás felé. A pontosság és a ms ismeretlen. A négyosztályos JNET az élő appban továbbra is scope-on kívül van. | Magas. Klasszifikátor-tanító notebook a publikus fában nincs. | A TDK a szegmentáló A/B és a mérési protokoll helyett második tanításba csúszik. | A választott architektúra (torchvision vagy timm) licence a döntéskor rögzítendő. A kapott súly privát marad. |
| Csomagolás és latenciamérés | PyInstaller onedir + opcionális Inno Setup. Napló: `performance_log.csv`. | Nem új csomagoló. Ugyanaz az onedir, plusz jegyzőkönyv: gép, CPU/GPU, backend, `Seg_Model`, `Cls_Model`, a CSV oszlopai. ORT csak akkor kerül a spec-be, ha a ViT ONNX A/B megvan. | A dolgozat állítása a naplóból jön. A ViT sorban a szegmentálás ideje 0; a ResNet sorban a szegmentálás és a Keras is benne van. A két utat külön kell kiírni. | Alacsony a jegyzőkönyvre. Közepes, ha az ORT bekerül a buildbe. | A `MODELLOK_MASOLASA.txt` ViT sora hiányzik. A SQLite `cls_model` és a CSV felirata ViT mentésnél eltér. Ezek a demót félrevihetik, a modell minőségét nem mérik. | A repó MIT (Copyright 2020 Gongfan Fang, upstream DeepLab). A becsomagolt PyTorch és TensorFlow wheel a saját licence. |

## 3. Külön fókusz: LLM

Az LLM a TDK-ban narrátor lehet a már kiszámolt mezők fölött. Klinikai indoklást a modell nem kap feladatul. A bináris fej nem tud 2A / 2B / 3-at; egy szöveg, ami ezt mégis kiírja, hamis eredmény.

### Két út

**A — szöveg a strukturált kimeneten (ezt érdemes TDK-ra vinni).** A kép és a maszk a gépen marad. A prompt csak mezőket kap, például: backend neve (`ViT-B/16` vagy `ResNet50V2`), `p1` vagy a Keras sigmoid, a 0,5 küszöb, a GUI címke, ResNet úton a maszk területe és a bbox, `Seg_Model`. A modell egy rövid magyar mondatot ad, ami ezeket a mezőket mondja vissza, fix döntéstámogató zárlattal. Tool-calling vagy egy kitöltött sablon ugyanazt a célt éri. Ha a mezők üresek, a mondat is üres marad (`Nincs polip`, `Classifier unavailable`).

**B — multimodális VLM a frame-en és a maszkon (álom a beadandóhoz).** MedGemma 1.5 4B multimodális a [model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card) szerint (frissítés: 2026-01-13; bemenet szöveg és kép, kimenet szöveg). A súly HAI-DEF feltételű, nyílt súly, nem OSI értelemben nyílt forrás. A kísérő GitHub-kód Apache-2.0. A kártya fejlesztői kiindulópontnak szánja, önálló orvosi alkalmazásnak nem. Erre az útra a repóban nincs finomhangoló notebook, nincs instrukciós pár, és nincs latencianapló.

A TDK-szöveg az A utat mutathatja demón, a B-t csak határként: mi kellene hozzá, és miért nem fér bele ebbe a fordulóba.

### Adat

- Az A úthoz új címke nem kell. A mezők a meglévő inferenciából jönnek.
- A B úthoz kép–maszk–szöveg párok kellenének. Ilyen szöveg a fában nincs. Klinikai leletből gyártani PHI-t csinálna.
- A tanító Excel (`images/PI-program-JNET-classes.xlsx`) és a képek továbbra is a gépen maradnak. LLM-notebookba bemásolni őket tilos.
- A modell súlya (`*.pth`, MedGemma, ONNX) a `.gitignore` alatt marad.

### Adatvédelem

- PHI, klinikai kép, betegazonosító és lelet a publikus repóba, a commitba, a notebook kimenetébe és a prompt példába nem kerül.
- Felhős API-nak klinikai frame nem megy. Ha egyáltalán fut modell, az helyi.
- Az A út promptja akkor tiszta, ha pixel nincs benne. A `p1` és a maszkterület önmagában is érzékeny lehet, ha mellé ID kerül; a példa prompt ID nélkül készül.
- A B út a pixelt is a modellbe adja. Az ugyanaz a kezelés, mint a klinikai képé: helyi lemez, nincs GitHub, nincs képernyőfotó a dolgozat nyilvános mellékletében.

### Latencia

A `<1 s` a gombnyomásos képi út célja (előfeldolgozás + a futó backend), nem mért eredmény. A repóban nincs rögzített ms. A `performance_log.csv` oszlopai megvannak: `Total_Time_ms`, `Preprocess_ms`, `Segmentation_Inference_ms`, `Postprocess_ms`, `Classification_ms`. ViT sorban a szegmentálás 0. ResNet sorban a szegmentálás és a Keras együtt számít.

Az LLM külön fázis a képi eredmény után, vagy a demón előre kiszámolt mondat. Egy 4B multimodális MedGemma a hallgatói laptopon a képi út mellé, ugyanabba az 1 másodpercbe, nem vállalható: mérés nincs, a modell a fában nincs, és a vision encoder a strukturált JSON-nál nagyobb munka. A pontos ms ismeretlen, amíg valaki helyi gépen nem méri, klinikai kép nélkül, nyilvános vagy szintetikus frame-en.

### Egyetem és termék

| | TDK / kutatás | Termék |
|--|----------------|--------|
| HarDNet-MSEG | Apache-2.0, hivatkozással | Ugyanaz, ha a vendored `LICENSE` a fában marad |
| PraNet-V2 | A MIT `LICENSE` kutatásra elég, hivatkozással | V1 kereskedelmi korlát: előbb diff, kétség esetén engedély |
| ONNX Runtime | MIT, saját export | Tiszta út, a súly továbbra is privát |
| TensorRT | Egyetemi gépen, az NVIDIA feltétellel | A SDK licence külön; a repó MIT-je nem elég |
| MedGemma | HAI-DEF feltételek, fejlesztői prototípus, döntéstámogató mondat | Tilos úgy használni, hogy egy egészségügyi hatóság a Google-t az eszköz gyártójának minősítse. Önálló diagnosztikai terméknek a kártya nem szánja. A tiltott felhasználás a HAI-DEF policy |

A szöveges A útnál a konkrét kis modell (Gemma, Llama, Qwen vagy más) licence a választáskor egy sor a dolgozatban. Most nincs kiválasztott modell. Kereskedelmi és kutatási feltételeik eltérnek; a repó MIT-je őket nem fedi.

### Megfogalmazás a dolgozatban

A mondat a mezőket ismétli. Javasolt zárás, szó szerint a UI közelében: az eredmény döntéstámogatás, nem diagnózis. A modell nem emeli a bináris fejet JNET alosztályra. Nem ír terápiát. Ha a klasszifikátor nem töltődött, a szöveg is ezt mondja.

## 4. Prioritás TDK-ig

### Must

- Mérési jegyzőkönyv a biblia 8. pontja szerint, szám nélkül addig, amíg a mérés megvan. Ugyanaz a patient-ID split. Szegmentálás: a notebook IoU, HarDNetnál Dice és IoU. Klasszifikáció: accuracy, sensitivity, specificity, backendenként külön.
- HarDNet-MSEG végigvitele lokálisan a #3 notebookon, a DeepLab epoch 31 és a U-Net epoch 22 ellen. A súly a `checkpoints_hardnet/` alatt marad, nem a publikus fában. A #3 addig draft.
- Laptop baseline a `performance_log.csv`-ből: ViT út (`Seg_Model = none`) és ResNet + szegmentáló út külön. A gép és a CUDA/CPU fel van írva.
- A dolgozat és a demó szövege döntéstámogatás. PHI nincs a repóban, a promptban és a mellékletben.
- Egy checkpointnév a DeepLabnál: az app `best_model_0409_epoch_31.pth`. Az epoch 38 csak akkor kerül a szövegbe, ha azt a fájlt ténylegesen mérik, és ez a jegyzőkönyvben szerepel.

### Should

- ViT ONNX export és ONNX Runtime A/B ugyanazon a laptopon, pontosság és `Classification_ms`. A Keras gráf nincs ebben a feladatban.
- Demó előtt a napló őszinte: a SQLite `cls_model` az `active_cls_model_name()` legyen, és a `MODELLOK_MASOLASA.txt` kapjon ViT sort, ha az exe-t viszik.
- Egy rögzített JSON-séma az A út mezőire, modellhívás nélkül is. Ebből vagy sablonmondat lesz, vagy később egy kis helyi modell.
- A TDK egy szegmentáló jelöltet mérjen végig. A PraNet-V2 addig nem indul, amíg a HarDNet táblázat üres.

### Stretch

- FP16 vagy INT8, illetve TensorRT, ha az ONNX A/B megvan és a pontosságdeltát Levente látta.
- Könnyebb klasszifikátor saját tanítással, ha az ONNX ViT a kereten kívül marad. Publikus idegen súly nélkül.
- PraNet-V2 második szegmentáló, a V1 diff után.
- Helyi kis szövegmodell az A út JSON-jára, offline, pixel nélkül a promptban.
- MedGemma frame + maszk csak külön gépen, HAI-DEF mellett, a `<1 s` élő úton kívül. Finomhangolás ebbe a fordulóba nem fér: nincs instrukciós készlet a fában.

## 5. Realisztikus és álom-max

### Realisztikus

Egy szegmentáló jelölt, a HarDNet-MSEG, végigmérve a meglévő patient-ID spliten a két jelenlegi checkpoint ellen. A klasszifikátor a ViT-B/16 marad. A gyorsítás saját ONNX export, ha a baseline ms után még kell. A laptopon a `performance_log.csv` a bizonyíték. Az LLM egy sablon vagy egy kis helyi modell, ami a már kiszámolt mezőket mondja magyarul, döntéstámogató zárlattal. A csomag a mai PyInstaller onedir. A #2 és a #3 draft marad, amíg Levente mást nem mond. Súly, kép és PHI a repón kívül marad.

### Álom-max

HarDNet és PraNet-V2 is megvan a saját spliten, kvantált TensorRT motorban, egy könnyebb, újratanított klasszifikátorral, és a teljes gombnyomás 1 másodperc alatt marad egy multimodális MedGemmával együtt, ami a frame-ből JNET-indoklást ír. Ehhez nincs mérés, nincs finomhangoló adat, a négyosztályos fej az élő appban nincs, és a HAI-DEF meg a PraNet-V1 kereskedelmi szövege ezt termékállításként nem viseli. Ez a felső határ a beszélgetéshez. A beadandó a realisztikus lista.
