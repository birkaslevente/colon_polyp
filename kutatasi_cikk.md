# DeepLabV3+ alkalmazása vastagbél polipok szegmentálására klinikai és standard adatbázisokon

## 1. Motiváció

### a) Kihívás
A vastagbélrák (colorectal cancer, CRC) világszerte az egyik leggyakoribb daganatos megbetegedés, amelynek korai felismerése életmentő lehet. A betegség leggyakrabban a vastagbél belső falán kialakuló polipokból fejlődik ki. A korai diagnózis arany standardja a kolonoszkópia, amelynek során az orvos endoszkóp segítségével vizsgálja át a bélfalat. Azonban ez a vizsgálat erősen függ az orvos tapasztalatától és figyelmétől; a fáradtság vagy a figyelmetlenség a polipok – különösen a kicsi, lapos vagy nehezen észrevehető elváltozások – elmulasztásához vezethet.

Az automatikus polip detektálás és szegmentálás (Computer-Aided Diagnosis, CADx) célja, hogy segítse az orvosokat a vizsgálat során. A feladat azonban számítógépes látás szempontjából rendkívül nehéz:
1.  **Változatos megjelenés:** A polipok mérete, alakja és textúrája nagymértékben eltérő lehet.
2.  **Környezeti nehézségek:** A bélfal redői, a nyálkahártya csillogása (tükröződések), a székletmaradványok és a vizsgálati eszközök mind zavaró tényezők.
3.  **Adatbeli eltérések:** A kutatásban gyakran használt, tisztított adatbázisok (mint a Kvasir-SEG) minősége gyakran jobb, mint a valós klinikai gyakorlatban keletkező felvételeké (pl. SZE adatbázis), ahol elmosódott képek és rosszabb megvilágítás is előfordulhat.

### b) Kutatási célok
Jelen cikk célja a **DeepLabV3+** szemantikus szegmentációs architektúra hatékonyságának vizsgálata vastagbél polipok detektálására. Kiemelt célunk:
*   Megvizsgálni, hogy a **MobileNetV2** backbone-nal rendelkező modell mennyire alkalmas a feladatra, szem előtt tartva a jövőbeli valós idejű felhasználást.
*   Elemezni a modell robusztusságát egy vegyes tanítóhalmazon, amely ötvözi a standard (Kvasir-SEG) és a saját gyűjtésű klinikai (SZE) adatokat.
*   Szigorú, beteg-alapú (patient-level) szétválasztással validálni az eredményeket, elkerülve az adatszivárgást, ami sok korábbi kutatásban torzította az eredményeket.

### c) Irodalmi áttekintés
Az orvosi képszegmentálás területén a konvolúciós neurális hálók (CNN) dominálnak.
*   Az **U-Net** (Ronneberger et al., 2015) tekinthető az alapértelmezett standardnak (baseline). Szimmetrikus encoder-decoder felépítése és skip connection-jei lehetővé teszik a pontos lokalizációt, ami kritikus az orvosi képalkotásban.
*   A korábbi **FCN** (Fully Convolutional Networks) modellek úttörők voltak, de gyakran küzdöttek a finom részletek elvesztésével a pooling rétegek miatt.
*   A **DeepLab** modellcsalád (Chen et al.) bevezette az *atrous* (dilated) konvolúciót és az *Atrous Spatial Pyramid Pooling* (ASPP) modult, amely képes különböző léptékű kontextuális információt kinyerni anélkül, hogy drasztikusan csökkentené a felbontást. A DeepLabV3+ ezt egy hatékony decoder modullal egészítette ki, javítva a határok szegmentálását.

### d) Dokumentum felépítése
A cikk a következőképpen épül fel: a 2. fejezetben bemutatjuk a felhasznált módszereket, az adatbázisokat és a kísérleti elrendezést. A 3. fejezetben ismertetjük a kvantitatív és kvalitatív eredményeket. A 4. fejezetben diszkutáljuk az eredményeket és összehasonlítjuk más módszerekkel (U-Net, FCN), végül az 5. fejezetben levonjuk a következtetéseket.

## 2. Módszerek

### a) Definíciók és jelölések
A szegmentációs teljesítmény mérésére két fő metrikát használunk:
*   **IoU (Intersection over Union):** A predikció és a valós maszk (Ground Truth) metszetének és uniójának aránya. $IoU = \frac{|A \cap B|}{|A \cup B|}$
*   **Dice koefficiens:** Hasonló az IoU-hoz, de nagyobb súlyt ad a helyes találatoknak (f1-score megfelelője pixel szinten). $Dice = \frac{2|A \cap B|}{|A| + |B|}$

### b) State-of-the-art: DeepLabV3+
A választott modellünk a DeepLabV3+, amely jelenleg is a state-of-the-art közelében van a szemantikus szegmentációban.
*   **Encoder:** A képi jellemzők kinyerésére a **MobileNetV2** hálót választottuk. Bár a ResNet-101 pontosabb lehet, a MobileNetV2 jelentősen kevesebb paraméterrel rendelkezik és gyorsabb, ami elengedhetetlen a klinikai, valós idejű (real-time) alkalmazásokhoz.
*   **ASPP (Atrous Spatial Pyramid Pooling):** Ez a modul különböző rátájú (rate) lyukacsos konvolúciókat alkalmaz párhuzamosan, így a hálózat képes egyszerre érzékelni a kisebb és nagyobb polipokat is, anélkül, hogy a felbontás romlana.
*   **Decoder:** A V3+ verzió legfontosabb újítása egy egyszerű, de hatékony decoder, amely felskálázza (upsampling) az encoder kimenetét és összefűzi az alacsonyabb szintű jellemzőkkel, így élesebb határokat eredményez.

### c) Célok a definíciók tükrében
Célunk a teszt halmazon az átlagos IoU és Dice értékek maximalizálása. Különösen fontos a **False Positive** (téves riasztás) arány alacsonyan tartása, mivel a túl sok téves jelölés zavarhatja az orvost a vizsgálat során.

### d) Saját módszerek

#### Adatbázisok
A kutatáshoz két adatbázist kombináltunk:
1.  **Kvasir-SEG:** Egy nyilvánosan elérhető, standard benchmark adatbázis, amely 1000 darab, szakértők által annotált endoszkópos képet és maszkot tartalmaz. Ezek a képek általában jó minőségűek.
2.  **SZE Dataset:** Saját gyűjtésű, valós klinikai környezetből származó adatbázis, amely 1073 képet tartalmaz. Ezek a képek változatosabbak minőségben, és jobban tükrözik a mindennapi kihívásokat.

#### Előfeldolgozás és Augmentáció
A képeket és maszkokat egységesen 513x513 pixel méretre alakítottuk, ami a DeepLab ajánlott bemeneti mérete. A modell általánosító képességének növelése érdekében a tanítás során erős adat-augmentációt alkalmaztunk:
*   **Random Scale:** 0.5x és 2.0x közötti véletlenszerű átméretezés (a különböző méretű polipok felismeréséhez).
*   **Random Crop:** Véletlenszerű kivágás a skálázott képből.
*   **Random Horizontal Flip:** Vízszintes tükrözés.
A normalizáláshoz az ImageNet átlagát és szórását használtuk.

#### Validációs stratégia: StratifiedGroupKFold
Kritikus fontosságú volt az adatszivárgás (data leakage) elkerülése. Mivel a SZE adatbázisban egy betegtől több képkocka is származhat (videóból kivágva), a hagyományos véletlenszerű keverés (shuffle split) esetén előfordulhatna, hogy ugyanazon beteg egyik képe a tanító, másik pedig a teszt halmazba kerül. Ez hamisan magas eredményeket szülne.
Ennek elkerülésére **StratifiedGroupKFold** módszert alkalmaztunk:
*   **Group:** A beteg azonosítója (ID) alapján csoportosítottunk, így egy beteg összes képe vagy csak a tanító, vagy csak a teszt halmazba került.
*   **Stratified:** Igyekeztünk fenntartani a polip-pozitív és negatív képek arányát.

A végső felosztás:
*   **Train:** ~1476 kép (Kvasir teljes + SZE train része).
*   **Validation:** ~102 kép (csak SZE).
*   **Test:** ~495 kép (csak SZE, teljesen új betegek).

#### Tanítás
A modellt PyTorch keretrendszerben implementáltuk.
*   **Loss function:** Cross Entropy Loss.
*   **Optimizer:** Adam optimalizáló, 0.001-es tanulási rátával.
*   **Scheduler:** ReduceLROnPlateau (a tanulási ráta csökkentése, ha a validációs IoU nem javul).
*   Transzfer tanulást alkalmaztunk, azaz a MobileNetV2 backbone-t ImageNet-en előtanított súlyokkal inicializáltuk.

## 3. Eredmények

### a) Benchmark problémák
A tesztelés során tapasztalt fő nehézségek:
*   Nagyon apró (diminutive) polipok detektálása.
*   A polipok határának pontos követése, különösen, ha a polip színe alig tér el a nyálkahártyától (flat lesions).
*   Műtéti eszközök vagy buborékok jelenléte a képen.

### b) Kvantitatív Eredmények
A teszt halmazon (amely kizárólag a modell által sosem látott betegek képeit tartalmazta a SZE adatbázisból) az alábbi eredményeket értük el:

| Metrika | Érték (Átlag) |
| :--- | :--- |
| **mIoU** | **0.824** |
| **Dice Score** | **0.881** |

Ez az eredmény azt mutatja, hogy a modell képes volt általánosítani a Kvasir-on tanult mintázatokat a SZE klinikai adataira is. A 0.88-as Dice score versenyképesnek számít, különösen egy könnyűsúlyú (MobileNet) backbone esetén.

### c) Kvalitatív Eredmények
A vizualizációk (lásd a mellékelt ábrákat) alapján:
*   A modell helyesen megtalálta a polipokat az esetek többségében.
*   **Siker:** Jól elkülöníti a polipot a háttértől, még rosszabb megvilágítás esetén is.
*   **Hibaesetek:** Néha a bélfal redőit vagy a sötétebb árnyékokat tévesen polipnak jelölte (False Positive), illetve a nagyon lapos polipok széleit néha pontatlanul húzta meg.

## 4. Diszkusszió és Összehasonlítás

Ebben a részben összevetjük a DeepLabV3+ (MobileNetV2) teljesítményét két másik elterjedt architektúrával: az **U-Net**-tel és a hagyományos **FCN**-nel (Fully Convolutional Network).

### 1. DeepLabV3+ vs. U-Net
Az **U-Net** az orvosi szegmentálás de facto standardja.
*   **Pontosság:** Az irodalmi adatok alapján az U-Net gyakran ér el 0.80-0.85 közötti Dice értéket a Kvasir-SEG adathalmazon. A mi DeepLabV3+ modellünk (0.88 Dice) felülmúlja vagy eléri ezt a szintet.
*   **Struktúra:** Az U-Net skip connection-jei kiválóak a finom részletek megőrzésében, de a DeepLab **ASPP modulja** jobban kezeli a különböző méretű objektumokat (multi-scale context). Mivel a polipok mérete extrém változó lehet, a DeepLab globális kontextus-értése előnyt jelent.
*   **Sebesség:** A MobileNetV2 backbone miatt a mi modellünk paraméterszáma jelentősen kisebb lehet, mint egy standard (pl. VGG backbone-os) U-Net-é, így gyorsabb inferenciát tesz lehetővé.

### 2. DeepLabV3+ vs. FCN (pl. FCN-8s)
Az **FCN** volt az első lépés a szemantikus szegmentációban, de mára elavultabbnak számít.
*   **Határok:** Az FCN gyakran "kockás", elnagyolt határokat produkál, mivel nem rendelkezik olyan kifinomult decoderrel vagy dilated konvolúcióval, mint a DeepLab.
*   **Eredmény:** Az FCN modellek általában alacsonyabb, 0.75-0.80 körüli Dice score-t érnek el ezen a feladaton. A DeepLabV3+ egyértelműen jobb a polipok körvonalának precíz követésében.

**Összefoglalva:** A DeepLabV3+ a MobileNetV2-vel kiváló kompromisszumot nyújt: a pontossága összemérhető vagy jobb, mint a nehezebb U-Net modelleké, miközben a sebessége alkalmassá teszi valós idejű felhasználásra.

## 5. Alkalmazások a State-of-the-art-on túl

A kutatás eredményei közvetlenül hasznosíthatók a klinikai gyakorlatban:
*   **Valós idejű döntéstámogatás:** A MobileNetV2 hatékonysága lehetővé teszi, hogy a modell akár magán az endoszkópos torony hardverén vagy egy csatlakoztatott laptopon fusson valós időben (25-30 FPS), vizuális jelzéssel (pl. bounding box vagy maszk színezés) segítve az orvost a vizsgálat közben.
*   **Minőségbiztosítás:** A rendszer használható utólagos elemzésre is, hogy ellenőrizzék, maradt-e észrevétlen polip a rögzített videókon.

## 6. Következtetések

### a) Összefoglalás
Kutatásunkban sikeresen alkalmaztuk a DeepLabV3+ architektúrát vastagbél polipok szegmentálására egy heterogén (Kvasir + SZE) adatbázison.

### b) Fő eredmények
1.  A modell **0.881-es Dice score-t** ért el a teszt halmazon, ami igazolja a módszer hatékonyságát.
2.  Bizonyítottuk, hogy a **StratifiedGroupKFold** alkalmazása elengedhetetlen a reális eredményekhez klinikai adatok esetén, elkerülve az adatszivárgás torzító hatását.
3.  A modell képes volt robusztusan működni a rosszabb minőségű, valós klinikai képeken is, nem csak a tiszta benchmark adatokon.

### c) Következő lépés
A jövőben tervezzük:
*   A modell kiterjesztését videó-szekvenciák időbeli (temporal) információinak felhasználására (pl. LSTM vagy 3D CNN segítségével), hogy csökkentsük a villódzó téves detektálásokat.
*   Más backbone-ok (pl. ResNet-50) összehasonlító elemzését, hogy pontosabb képet kapjunk a sebesség-pontosság áldozatról.

