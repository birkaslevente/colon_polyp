# Szakdolgozat Vázlat: Valós idejű vastagbél polip detektálás és osztályozás támogatása mélytanulással

## 1. Bevezetés (Introduction)
*Becsült terjedelem: 2-3 oldal*

### 1.1. Motiváció és klinikai háttér
#### 1.1.1. A vastagbélrák népegészségügyi jelentősége
- A fejezet bemutatja a colorectalis rák (CRC) előfordulási statisztikáit (pl. WHO adatok).
- Halálozási arányok ismertetése.
- A korai felismerés (szűrés) fontosságának hangsúlyozása.

#### 1.1.2. A kolonoszkópia szerepe és az "elmulasztott polipok" (miss rate) problémája
- A kolonoszkópia mint "gold standard" vizsgálat leírása.
- Az emberi tényező (fáradtság, figyelemhiány) szerepe.
- Szakirodalmi hivatkozás a polipok 20-25%-os elmulasztási arányáról.

#### 1.1.3. A számítógépes diagnosztika (CADx) szükségessége a valós idejű vizsgálatok során
- Érvelés a valós idejű segédrendszer ("második szempár") mellett.
- Hogyan csökkentheti a hibákat és növelheti az adenoma detektálási arányt (ADR).

### 1.2. Technológiai kihívások
#### 1.2.1. Valós idejű képfeldolgozás követelményei
- A "valós idő" fogalmának definíciója ebben a kontextusban (min. 25 FPS, <100ms látencia).
- Cél: az orvost ne zavarja a késés.

#### 1.2.2. HDMI videójel feldolgozásának nehézségei orvosi környezetben
- A videójel digitalizálásának (capture kártya) technikai aspektusai.
- Felbontás-kezelés és a szinkronizáció fontossága.

#### 1.2.3. Különböző AI modellek integrációja
- Problémafelvetés: nehéz szegmentáló és klasszifikátor párhuzamos futtatása Pythonban.

### 1.3. A szakdolgozat célkitűzései
#### 1.3.1. DeepLabV3+ alapú szegmentáló modell fejlesztése és tanítása
- Konkrét cél: a modell illesztése a polip detektálási feladatra.

#### 1.3.2. Moduláris Python keretrendszer létrehozása HDMI jelfeldolgozásra
- Szoftverfejlesztés: "híd" az orvosi eszköz és az AI modellek között.

#### 1.3.3. Interfész biztosítása külső osztályozó modulok számára
- A "plug-and-play" architektúra koncepciója.

### 1.4. A dolgozat felépítése
- Rövid "útikalauz" a fejezetek tartalmáról.

## 2. Elméleti háttér és módszertan (Theoretical Background and Methods)
*Becsült terjedelem: 8-10 oldal*

### 2.1. Alapfogalmak és definíciók
#### 2.1.1. Szemantikus szegmentáció vs. objektum detektálás
- A különbség tisztázása: bounding box (detektálás) vs. pixel-szintű maszkolás (szegmentálás).
- Indoklás a szegmentáció választása mellett (pontosabb lokalizáció).

#### 2.1.2. Kiértékelési metrikák
- Matematikai definíciók (képletekkel): IoU (Jaccard), Dice score, Precision, Recall, FPS.

#### 2.1.3. Valós idejű rendszerek teljesítménymutatói
- Throughput (feldolgozott képkocka/mp) fogalma.
- Latency (bemenettől a megjelenítésig eltelt idő) fogalma.

### 2.2. Irodalmi áttekintés (State-of-the-art)
#### 2.2.1. Mélytanulás az orvosi képfeldolgozásban
- Történeti áttekintés: FCN, U-Net (az orvosi standard).
- Ezek korlátainak bemutatása.

#### 2.2.2. A DeepLab modellcsalád és az Atrous Spatial Pyramid Pooling (ASPP)
- A DeepLab újításainak részletes bemutatása (dilated convolution).
- ASPP szerepe a különböző méretű objektumok felismerésében a felbontás csökkenése nélkül.

#### 2.2.3. Valós idejű videófeldolgozás Python környezetben
- Python GIL (Global Interpreter Lock) korlátai.
- Threading vs multiprocessing.
- A GPU gyorsítás (CUDA) szerepe.

### 2.3. Alkalmazott eszközök és technológiák
#### 2.3.1. Programozási környezet
- Python 3.x.
- PyTorch (mint DL keretrendszer).

#### 2.3.2. Képfeldolgozó könyvtárak
- OpenCV (videó kezelés).
- NumPy (mátrix műveletek).
- Albumentations (augmentáció).

#### 2.3.3. Hardveres környezet
- A fejlesztéshez használt GPU (pl. NVIDIA RTX).
- A videó capture eszköz specifikációja.

## 3. A szegmentáló modell tervezése és implementációja (Segmentation Model Design)
*Becsült terjedelem: 6-8 oldal*

### 3.1. Adatbázisok és előkészítés
#### 3.1.1. Adatbázisok bemutatása
- Kvasir-SEG (jellemzők, méret).
- SZE Dataset (saját gyűjtés, klinikai sajátosságok).

#### 3.1.2. Adattisztítás és annotációs kihívások
- Hibás maszkok és üres képek kezelése.

#### 3.1.3. Augmentációs stratégiák
- Konkrét augmentációk felsorolása (RandomScale, Flip, Crop).
- Indoklás: a modell általánosító képességének növelése.

### 3.2. Modell architektúra
#### 3.2.1. DeepLabV3+ kiválasztásának indoklása
- Miért ezt választottuk az U-Net helyett (jobb határvonal-detektálás).

#### 3.2.2. MobileNetV2 backbone előnyei
- Összehasonlítás a ResNet-tel paraméterszám és sebesség tekintetében.
- Miért kritikus ez a valós idejű működéshez.

#### 3.2.3. Loss function és optimalizáció
- CrossEntropyLoss használata.
- Adam optimalizáló beállításai.
- Learning Rate Scheduler.

### 3.3. Tanítási és validációs stratégia
#### 3.3.1. Transzfer tanulás
- ImageNet-en előtanított súlyok használata a konvergencia gyorsítására.

#### 3.3.2. A StratifiedGroupKFold jelentősége
- Részletes magyarázat a beteg-alapú csoportosításról (Group K-Fold) az adatszivárgás elkerülésére.
- Összehasonlítás a sima random splittel.

#### 3.3.3. Hiperparaméterek
- Batch size, epoch szám.
- Input felbontás (513x513) megválasztása.

## 4. A szoftveralkalmazás tervezése és fejlesztése (Software Application Design)
*Becsült terjedelem: 6-8 oldal*

### 4.1. Rendszerarchitektúra és követelmények
#### 4.1.1. Funkcionális követelmények
- Stream fogadása.
- Szegmentálás.
- Maszk rávetítés.
- ROI kivágás.
- Osztályozás hívása.
- Eredmény kiírása.

#### 4.1.2. Nem-funkcionális követelmények
- Stabilitás (nem fagyhat le műtét közben).
- Válaszidő (< X ms).
- Bővíthetőség.

#### 4.1.3. Adatfolyam diagram
- Vizuális ábra és leírás az adatok útjáról a HDMI bemenettől a monitorig.

### 4.2. Implementációs részletek
#### 4.2.1. Videó rögzítő modul
- OpenCV VideoCapture használata.
- Pufferelés (Queue) a frame-eldobás elkerülésére aszinkron szálon.

#### 4.2.2. Inferáló motor
- A modell eval() módja.
- torch.no_grad().
- Tensor-Numpy konverziók optimalizálása a sebesség érdekében.

#### 4.2.3. A Klasszifikátor Interfész specifikációja
- API leírása: `classify(roi_image) -> prediction_class, confidence`
- Hogyan vágja ki a rendszer a polip területét (bounding box a maszk alapján).
- Átadás a "fekete doboz" osztályozónak.
- Dummy klasszifikátor implementációja a teszteléshez.

### 4.3. Felhasználói felület (GUI) terve
#### 4.3.1. Orvosi felhasználói élmény (UX)
- A zavaró tényezők minimalizálása (sötét téma, nem villogó feliratok).

#### 4.3.2. Vizualizációs megoldások
- Alpha blending (félig átlátszó maszk).
- Színkódolás (pl. piros=neoplasztikus, zöld=hiperplasztikus - az osztályozó kimenete alapján).

## 5. Eredmények és kiértékelés (Results and Evaluation)
*Becsült terjedelem: 4-5 oldal*

### 5.1. A szegmentáló modell teljesítménye
#### 5.1.1. Kvantitatív eredmények
- Táblázatos formában a SZE teszt halmazon elért IoU és Dice értékek.

#### 5.1.2. Kvalitatív elemzés
- Képernyőmentések a program működéséről: jó detektálás, részleges detektálás.
- Fals pozitív esetek (pl. széklet).

### 5.2. Rendszerteljesítmény mérése
#### 5.2.1. Feldolgozási sebesség (FPS)
- Mérések különböző felbontásokon (VGA, HD, Full HD).
- Hardver összehasonlítás (Laptop GPU vs. Desktop GPU).

#### 5.2.2. End-to-end késleltetés
- Mérés a kamera mozgása és a képernyőn megjelenő reakció között.

#### 5.2.3. Erőforrás-használat
- Memória (RAM/VRAM) és CPU terhelés grafikonok.

### 5.3. Diszkusszió
#### 5.3.1. A valós idejű működés korlátai
- Hol van a szűk keresztmetszet? (Adatmozgatás a GPU-ra, Python lassúsága, vagy maga a modell inferencia).

#### 5.3.2. Illeszthetőség a munkafolyamatba
- Mennyire zavaró vagy segítő a rendszer jelenlegi formájában.

## 6. Összefoglalás és jövőbeli tervek (Conclusions and Future Work)
*Becsült terjedelem: 1-2 oldal*

### 6.1. Az elért eredmények összefoglalása
- Sikeresen integrált DeepLabV3+ modell valós idejű Python környezetben.
- Működő osztályozó interfész.

### 6.2. A fejlesztés korlátai
- Jelenlegi FPS korlátok.
- Felbontásbeli kompromisszumok.

### 6.3. Jövőbeli fejlesztési irányok
#### 6.3.1. Hardveres optimalizáció
- TensorRT, C++ újraírás, vagy FPGA használata.

#### 6.3.2. Temporális utófeldolgozás
- A maszkok "simítása" időben, hogy ne vibráljon a detektálás.

#### 6.3.3. További integráció
- Kapcsolódás kórházi információs rendszerekhez (PACS).

## 7. Irodalomjegyzék (References)
- Felsorolás a felhasznált cikkekről, könyvekről és technikai dokumentációkról.
