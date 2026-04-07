# Melléklet Y – A Live Capture alkalmazás szoftveres dokumentációjának terve

## Terv a második melléklethez (az első melléklet struktúráját követve)

---

## 1. Cél és kontextus

### 1.1 A dokumentáció célja
Jelen melléklet a diplomamunkában bemutatott valós idejű polip szegmentáló alkalmazás (`live_capture_app.py`) szoftveres megvalósítását dokumentálja. Célja, hogy bemutassa az alkalmazás felépítését szoftvermérnöki szemszögből: a videófolyam kezelését, a szegmentáló modell integrációját és a felhasználói felület implementációját.

### 1.2 Kapcsolat a dolgozattal
Míg a dolgozat 4. fejezete (Szoftveralkalmazás) a rendszer követelményeit, az adatfolyam diagramot és a felhasználói élményt tárgyalja, ez a melléklet szigorúan az implementációra fókuszál: hogyan valósul meg a `VideoCaptureThread`, a `MedicalAnalyzer` és az `App` osztály, valamint a közöttük lévő adatáramlás.

### 1.3 Felhasználási szcenárió
Az alkalmazás élő videóforrásból (HDMI capture kártya, webkamera, videófájl) vagy szimulációs módban (képmappa) folyamatosan megjeleníti a képet, és a felhasználó gombnyomására (Space) vagy „Capture & Analyze” gombra igény szerint elvégzi a polip szegmentálást és a demonstrációs klasszifikációt. A kimenet három panelen jelenik meg: élő kép, hálózati bemenet, szegmentált eredmény.

---

## 2. Fejlesztési környezet és függőségek

### 2.1 Hardverkörnyezet
- **Bemeneti források:** HDMI capture kártya, USB webkamera, vagy helyi videófájl. Szimulációs módban: képmappa (`images/`).
- **Számítási egység:** CPU vagy GPU (CUDA); a modell automatikusan detektálja a rendelkezésre álló eszközt (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
- **Tesztelt konfiguráció:** ASUS Zenbook 14 (AMD Ryzen 7, integrált grafikus) – a 5. fejezetben bemutatott teljesítménymérések alapján.

### 2.2 Szoftverkörnyezet és verziók
- **Python:** 3.x
- **Kritikus függőségek:** `opencv-python` (cv2), `torch`, `torchvision`, `PIL` (Pillow), `numpy`, `tkinter` (GUI, általában Python része)
- **Projekt-specifikus:** `network.modeling` (deeplabv3plus_mobilenet), `utils.ext_transforms` (ExtResize, ExtCenterCrop, ExtToTensor, ExtNormalize)
- **Opcionális:** `pygrabber` (DirectShow kameralista Windows-on; hiányában egyszerű indexelés)

### 2.3 Környezet felállítása és futtatás
- A projekt gyökérkönyvtárából: `python live_capture_app.py`
- A checkpoint elérési útja: `checkpoints/best_model_epoch_38.pth` (konfigurálható a fájl elején)

---

## 3. Rendszerarchitektúra magas szinten

A rendszer három fő modulból áll, amelyek a dolgozat 4. fejezetének adatfolyam diagramjához igazodnak:

1. **VideoCaptureThread:** Dedikált szálon folyamatosan olvassa a videóforrást (kamera, videófájl) vagy szimulációs módban véletlenszerű képeket tölt be. A legfrissebb képkocka szálbiztos lock-kal védett.
2. **MedicalAnalyzer:** Betölti a DeepLabV3+ modellt, előfeldolgozza a képet (513×513, ImageNet normalizáció), inferenciát futtat, vizualizálja a maszkot és naplózza a teljesítményt (CSV).
3. **App (GUI):** Tkinter alapú hárompaneles felület, kezeli a forrásválasztást, a „Capture & Analyze” műveletet és a Space billentyű kötését.

---

## 4. Modulok részletesen

### 4.1 VideoCaptureThread
- **Felelősség:** Videófolyam vagy szimulációs képek folyamatos betöltése dedikált daemon szálon.
- **Bemenetek:** `src` (kamera index int vagy videófájl útvonal string), `simulation_mode`, `image_folder`.
- **Szimulációs mód:** `images/` mappa, szűrés `allowed_ids` lista alapján (pl. PI009, PI015, …). 2 másodpercenként véletlenszerű kép váltás.
- **Videófájl mód:** `set_position(percent)` – pozíció ugrás a sávval (slider). Hurok a fájl végén.
- **Kimenet:** `read()` – (ret, frame) tuple, szálbiztos.

### 4.2 MedicalAnalyzer
- **Felelősség:** Modell betöltése, előfeldolgozás, inferencia, vizualizáció, teljesítménynaplózás.
- **Modell inicializálás:** `deeplabv3plus_mobilenet(num_classes=2, output_stride=16)`. Checkpoint kulcsok: `model_state`, `model_state_dict`, `state_dict` vagy gyökér.
- **Előfeldolgozás:** `ExtResize(513)` → `ExtCenterCrop(513)` → `ExtToTensor` → `ExtNormalize` (ImageNet mean/std) – összhangban a notebook validációs transzformációival.
- **Inferencia:** `model(input_tensor)`, `output.max(1)[1]` → bináris maszk (513×513).
- **Vizualizáció:** Polip maszk alapján az eredeti kép tartalma megjelenik, háttér fekete.
- **Klasszifikáció:** Jelenleg `dummy_classify()` – véletlenszerű demonstrációs eredmény (Adenoma/Hyperplastic/Serrated). Interfész készen áll valódi modell cseréjére.
- **Naplózás:** `performance_log.csv` – Timestamp, Total_Time_ms, Preprocess_ms, Inference_ms, Postprocess_ms.

### 4.3 App (GUI)
- **Layout:** Grid: felső sor (forrásválasztó, gombok, Live Analysis checkbox, videó slider), középső sor (3 panel: Live Feed, Network Input, Polyp Segmentation), alsó sor (Capture & Analyze gomb, státusz).
- **Forrásválasztás:** Combobox – kameralista (`list_available_cameras()`), videófájlok a projekt mappájában. „Start Live Input” / „Simulation (Images)” gombok.
- **Billentyűkötés:** Space → `capture_and_analyze()`.
- **Live Analysis:** Checkbox bekapcsolásakor minden GUI frissítési ciklusban (33 ms) lefut az elemzés – nagy terhelés, főleg demonstrációs célra.
- **Panel méretek:** Megjelenítés 400 px magasságúra skálázva.

---

## 5. Konfiguráció és fájlstruktúra

### 5.1 Konfigurációs változók (a fájl elején)
- `PERFORMANCE_LOG_FILE` – CSV napló neve
- `CHECKPOINT_PATH` – modell súlyok útvonala
- `NUM_CLASSES`, `OUTPUT_STRIDE`, `INPUT_SIZE` (513)
- `DEVICE` – automatikus CUDA/CPU választás

### 5.2 Várt fájlstruktúra
```text
projekt_gyökér/
├── live_capture_app.py
├── network/
│   └── modeling.py          # deeplabv3plus_mobilenet
├── utils/
│   └── ext_transforms.py    # ExtResize, ExtCenterCrop, stb.
├── checkpoints/
│   └── best_model_epoch_38.pth
├── images/                  # Szimulációs mód (opcionális)
│   └── *.jpg, *.png
└── performance_log.csv       # Generált futás közben
```

---

## 6. Adatfolyam és szálkezelés

### 6.1 Szálmodell
- **Capture szál:** Daemon thread, folyamatosan olvassa a forrást, lock-kal védi a `frame` változót.
- **GUI szál (main):** `root.after(33, update_live_feed)` – 30 FPS megjelenítés. A `capture_and_analyze` szinkron módon fut a főszálon (blokkoló).
- **Konfliktusok elkerülése:** `VideoCaptureThread.set_position` lock-kal védett, hogy a seek ne ütközzön az olvasással.

### 6.2 Elemzési folyamat (Capture & Analyze)
1. `video_thread.read()` – pillanatnyi képkocka
2. `analyzer.analyze(frame)` – preprocess → inference → postprocess → vizualizáció
3. Panel frissítés (snapshot, result)
4. Státusz szöveg: klasszifikációs eredmény + feldolgozási idő

---

## 7. Ismert limitációk és jövőbeli bővítés

- **Klasszifikátor:** Jelenleg dummy; valódi patológiai modell integrálása szükséges a klinikai használathoz.
- **Teljesítmény:** ~350 ms válaszidő CPU-n (Zenbook); GPU-val vagy TensorRT optimalizálással javítható.
- **4K / nagy felbontás:** Középre vágás 513×513-re – részletesebb kontextus elveszhet.
- **Platform:** A `pygrabber` és a DirectShow kameralista Windows-specifikus; Linux/Mac más megoldást igényel.

---

## 8. Reprodukálhatóság és futtatási lépések

- **Checkpoint:** A notebookban tanított `best_model_epoch_38.pth` (vagy aktuális legjobb) másolása a `checkpoints/` mappába.
- **Futtatás:** `python live_capture_app.py` a projekt gyökéréből.
- **Szimulációs teszt:** `images/` mappa létrehozása, releváns páciens ID-jú képek másolása (pl. PI009, PI015, …).

---

## Összefoglalás – Az első melléklethez képest

| Első melléklet (notebook)     | Második melléklet (live_capture_app) |
|-------------------------------|--------------------------------------|
| Offline tanítás, Jupyter      | Valós idejű alkalmazás, Tkinter GUI  |
| Adatbetöltés, augmentáció     | Videó/kép betöltés, előfeldolgozás  |
| Modell tanítás, checkpoint    | Modell betöltés, inferencia          |
| Reprodukálhatóság (seed, LR)  | Futtatási lépések, konfiguráció      |
| Predikciós kódminta           | Teljes elemzési pipeline (analyze)   |

A terv jóváhagyása után a fenti struktúra alapján megírható a teljes melléklet szövege (Markdown vagy LaTeX formátumban).
