## Ötletek ábrákhoz és táblázatokhoz fejezetenként

### 1. fejezet – Bevezetés, motiváció

- **1.2 Motiváció és klinikai háttér – ábraötletek**
  - Világtérkép vagy oszlopdiagram a colorectalis rák incidenciájáról és mortalitásáról (forrás: WHO / GLOBOCAN).
  - Egyszerű sematikus ábra az emésztőrendszerről, kiemelve a vastagbelet.
- **1.2 Motiváció és klinikai háttér – táblázatötletek**
  - Táblázat 2–3 ország (pl. Magyarország, EU átlag, világátlag) összehasonlított CRC incidenciájával és halálozásával.
  - Táblázat a szakirodalomban jelzett „miss rate” értékekről (pl. 20–25%) különböző tanulmányok szerint.

### 2. fejezet – Elméleti háttér és módszertan

- **2.1 Alapfogalmak – ábraötletek**
  - Illusztráció ugyanarra a képre: (1) bounding box-os objektumdetektálás, (2) szemantikus szegmentációs maszk; jól látszik, hogy a maszk pontosabb alakot ad.
- **2.2 Irodalmi áttekintés – ábra/táblázatötletek**
  - Táblázat néhány releváns publikációról (szerző, év, modell típusa, adatbázis, elért Dice/IoU, valós idejű-e).
  - Sematikus összehasonlító ábra U-Net vs. DeepLabV3+ architektúrákról (nagyon leegyszerűsített blokkszintű rajz).

### 3. fejezet – A szegmentáló modell tervezése és implementációja

- **3.1 Adatbázisok és előkészítés – ábra/táblázatötletek**
  - Példa eredeti endoszkópos kép + hozzá tartozó Kvasir-SEG maszk (előtte–utána páros).
  - Táblázat az adatbázis összetételéről: SZE Dataset vs. Kvasir-SEG (képszám, páciensek száma, felbontástartomány).
  - Ábra a különböző augmentációk hatásáról (eredeti kép, nagyított/forgatott, tükrözött változat).
- **3.2 Modell architektúra – ábraötletek**
  - Blokkvázlat a DeepLabV3+ MobileNetV2 architektúráról (encoder, ASPP, decoder, kimeneti maszk).
  - Kiemelő ábra az ASPP modulról (különböző atrous rate-ekkel).
- **3.3 Tanítási és validációs stratégia – ábra/táblázatötletek**
  - Tanítási és validációs IoU/Dice görbék epochok szerint (line chart).
  - Táblázat a fő hiperparaméterekről: batch size, epochok száma, learning rate, optimizer, scheduler.

### 4. fejezet – A szoftveralkalmazás tervezése és fejlesztése

- **4.1 Rendszerarchitektúra – ábraötletek**
  - Rendszerarchitektúra-diagram: HDMI forrás / kamera → capture kártya → Python alkalmazás → modell → vizualizáció / GUI.
  - Adatfolyam-diagram (Data Flow) az alkalmazáson belüli komponensekkel (VideoCaptureThread, MedicalAnalyzer, GUI).
- **4.2 Felhasználói felület és UX – ábraötletek**
  - Képernyőkép a Tkinter GUI-ról: élő kép panel, „Capture & Analyze” gomb, szegmentált eredmény, klasszifikációs címke.
  - Példa eset: ugyanaz a frame élő nézetben, bemeneti cropként és szegmentált maszk formájában egymás mellett.

### 5. fejezet – Eredmények és kiértékelés

- **5.1 A szegmentáló modell teljesítménye – ábra/táblázatötletek**
  - Oszlopdiagram az egyes teszthalmazon mért metrikákról (IoU, Dice, Precision, Recall).
  - Táblázat az SZE Dataset és (ha van) Kvasir-SEG-en mért eredményekről külön-külön.
  - Kvalitatív ábrák: több-példás montázs „nehéz esetekről” (kicsi polip, részben rejtett polip, tükröződés stb.) az eredeti képpel és a modell maszkjával.
- **5.2 Rendszerteljesítmény – ábra/táblázatötletek**
  - Diagram az FPS és a batch size / felbontás függvényében.
  - Oszlopdiagram az end-to-end válaszidő eloszlásáról (pl. átlag, szórás, percentilisek).
  - Táblázat a CPU/GPU terhelésről és memóriahasználatról különböző beállítások mellett.

### 6. fejezet – Diszkusszió és jövőbeli munka

- **6.1 Esettanulmány jellegű ábrák**
  - 1–2 olyan teljes vizsgálat kronologikus képsorozata, ahol a modell valósan segítette volna az orvost (pl. nehezen észrevehető kisméretű polip kiemelése).
- **6.2 Jövőbeli fejlesztési irányok – ábraötletek**
  - Sematikus ábra egy lehetséges jövőbeli rendszer-architektúráról (pl. felhőalapú kiértékelés, több modell együttműködése, automatikus riportgenerálás).

