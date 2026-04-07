# SZE Informatikai szakdolgozat checklist – ellenőrzés

Összevetés a jelenlegi `szakdolgozat.tex` és `irodalom.bib` állapotával.

---

## 1. Tárgyilagos, személytelen megfogalmazás

**Követelmény:** Pl. „a vizsgálat során megállapítható”, ne „megállapítottam”.

**Állapot:** **Nem teljesül.**

A szövegben sok első személyű, személyes megfogalmazás van, pl.:
- „választottam”, „alkalmaztam”, „használtam”, „implementáltam”, „megterveztem”
- „célkitűzésem”, „Szakdolgozatomban … megterveztem és implementáltam”
- „tartottam szem előtt”, „alkalmaztam” (sötét téma)

**Javaslat:** Átfogalmazni tárgyilagosra (passzív vagy „a dolgozatban … alkalmazásra került”, „a vizsgálat során megállapítható”). A bevezetés célkitűzésénél megengedhető a „célul tűztem ki”, de a többi fejezetben inkább személytelen stílus.

---

## 2. Szakszerű és pontos terminológia

**Követelmény:** Informatikai szakzsargon helyes magyar megfelelői.

**Állapot:** **Nagyjából megfelel.**

Használt kifejezések: szegmentálás, inferencia, backbone, adatszivárgás, validáció, augmentáció, ROI, stb. A szövegben angol kifejezések zárójelben (pl. „gerincét (backbone)”). Érdemes végigellenőrizni, hogy minden szakmai kifejezésnek legyen konzisztens magyar megfelelője, ahol az SZE elvárja.

---

## 3. Következetes rövidítéshasználat és rövidítésjegyzék

**Követelmény:** Pl. JSON, API, TCP/IP, SQL; szükség esetén rövidítésjegyzék.

**Állapot:** **Részben megfelel.**

A szövegben előfordulnak rövidítések (FPS, GPU, CPU, CADx, ROI, ADR, IoU, ASPP, GUI, UX, stb.). Nincs külön **rövidítésjegyzék** (az első előforduláskor zárójelben megoldva van). Ha a Tanszék kéri a jegyzéket, érdemes a dolgozat elején (pl. a tartalomjegyzék után) egy „Rövidítések” részt bevezetni és minden rövidítést ott definiálni.

---

## 4. IEEE hivatkozási stílus

**Követelmény:** IEEE stílus az irodalomjegyzékben és a szövegközi hivatkozásoknál.

**Állapot:** **Nem teljesül.**

- **Biblatex:** Jelenleg `\usepackage[backend=biber]{biblatex}` – **nincs** `style=ieee`. Alapértelmezetten más (ált. numerikus) stílus lesz.
- **Hivatkozások a szövegben:** A `.tex`-ben **egyetlen** `\cite{}` / `\parencite{}` / `\textcite{}` **nincs**. Az irodalom megvan a `.bib`-ben, de a szöveg nem hivatkozik rájuk.

**Javaslat:**
1. Preamble: `\usepackage[backend=biber, style=ieee]{biblatex}` (vagy a Tanszék által előírt pontos IEEE-biblatex opció).
2. A szövegben minden állításnál, ahol irodalomra támaszkodsz, tegyél be hivatkozást, pl. `\cite{deeplabv3plus}`, `\cite{kvasir_seg}`.
3. Fordítás után ellenőrizd, hogy a generált irodalomjegyzék formátuma megfelel-e a SZE IEEE követelményének.

---

## 5. Helyesírási szabályok

**Követelmény:** Különösen az összetett informatikai szavaknál.

**Állapot:** **Nincs ebben az ellenőrzésben nyelvtanilag átnézve.**

Érdemes külön (pl. helyesírás-ellenőrzővel vagy konzulenssel) végigvenni az összetett szavakat (pl. „valós idejű”, „adatfolyam”, „képfeldolgozás”, „felhasználói” stb.). A LaTeX-ben látható szövegeknek nincs nyilvánvaló helyesírási hibája.

---

## 6. Világos, logikus fejezetszerkezet

**Követelmény:** Bevezetés, Célkitűzés, Irodalmi áttekintés, Saját munka, Eredmények, Összegzés.

**Állapot:** **Megfelel.**

- 1. Bevezetés (motiváció, technológiai kihívások, **célkitűzések**, felépítés)
- 2. Elméleti háttér és módszertan (alapfogalmak, **irodalmi áttekintés**, eszközök)
- 3. Szegmentáló modell tervezése (**saját munka** – adat, modell, tanítás)
- 4. Szoftveralkalmazás (**saját munka** – követelmények, implementáció, GUI)
- 5. Eredmények és kiértékelés
- 6. Összefoglalás és jövőbeli tervek

A struktúra logikus és illeszkedik a SZE elvárásaihoz.

---

## 7. Magyar kivonat és angol Abstract (max. 1–1 oldal)

**Követelmény:** Magyar nyelvű kivonat és angol nyelvű Abstract.

**Állapot:** **Nem teljesül.**

A dokumentumban **nincs** `\begin{abstract} ... \end{abstract}` vagy külön „Kivonat” / „Abstract” fejezet. A tartalomjegyzék után közvetlenül a Bevezetés kezdődik.

**Javaslat:** A tartalomjegyzék után, a Bevezetés elé illeszd be:
- egy **magyar** „Kivonat” részt (max. 1 oldal), és
- egy **angol** „Abstract” részt (max. 1 oldal).

---

## 8. Formai követelmények: betűtípus, sorköz, igazítás

**Követelmény:** Times New Roman, 12 pt, 1,5-es sorköz, sorkizárt igazítás.

**Állapot:** **Részben megfelel.**

| Követelmény        | Jelenlegi állapot |
|--------------------|--------------------|
| Times New Roman    | Megvan: `\setmainfont{Times New Roman}` |
| 12 pt              | Megvan: `\documentclass[12pt, a4paper]{report}` |
| 1,5-es sorköz      | **Nincs beállítva.** Van `\usepackage{setspace}`, de **nincs** `\onehalfspacing` vagy `\setstretch{1.5}` a dokumentum elején. Alapértelmezett a single spacing. |
| Sorkizárt igazítás | A LaTeX report osztály alapértelmezetten sorkizárt (justify), ez rendben van. |

**Javaslat:** A `\begin{document}` után, a címlap előtt (vagy a címlap és a tartalomjegyzék után) add hozzá: `\onehalfspacing`. Ha a címlapot egyedi sorközzel szeretnéd, a címlap blokk után írd: `\onehalfspacing`, majd folytatódik a tartalom.

---

## 9. Margók: bal 3 cm (kötés), jobb / alsó / felső 2,5 cm

**Követelmény:** Bal 3 cm, jobb/alsó/felső 2,5 cm.

**Állapot:** **Nem teljesül.**

Jelenleg: `\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}`  
– a **bal margó 2,5 cm**, a követelmény **3 cm** (kötés miatt).

**Javaslat:** Cseréld így: `left=3cm` (a többi maradhat: top=2.5cm, bottom=2.5cm, right=2.5cm).

---

## 10. Ábrák és táblázatok sorszámozása és hivatkozása

**Követelmény:** Ábrák és táblázatok pontos sorszámozása és szöveges hivatkozása.

**Állapot:** **Részben / nem teljesül.**

- A szövegben **nincs** `\begin{figure}` vagy `\begin{table}`; tehát jelenleg **nincs számozott ábra vagy táblázat**.
- A 4.1.3 „Adatfolyam diagram” szekció **szövegesen** utal egy ábrára („megterveztem az adatfolyam … diagramot”, „Az ábra részletesen bemutatja”), de a dokumentumban **nincs beszúrt ábra** (nincs `\includegraphics` vagy beágyazott diagram).

**Javaslat:**
- Hozz létre és szúrj be minden fontos ábrát/táblázatot `figure` / `table` környezetben, címkével és sorszámmal.
- A szövegben minden ilyen ábrát/táblázatot hivatkozással említs meg (pl. „… az 1. ábra mutatja”, „1. táblázat”).

---

## 11. Forráskód fix szélességű betűtípussal

**Követelmény:** Courier New vagy Consolas (monospace).

**Állapot:** **Részben megfelel.**

A szövegben a kódrészleteknél használt `\texttt{...}` (pl. `\texttt{eval()}`, `\texttt{torch.no\_grad()}`) monospace megjelenést ad (a dokumentum alapbetűtípusa mellett a `\texttt` általában monospace). Ha a dokumentumban **hosszabb forráskód** is szerepel (több sor), érdemes `listings` vagy `verbatim` környezetben beilleszteni, és a listingsnél beállítani a fix szélességű betűtípust (pl. Consolas / Courier New), hogy teljes mértékben megfeleljen a követelménynek.

---

## 12. UML vagy rendszerszintű modellek a tervezési részben

**Követelmény:** UML diagramok vagy egyéb rendszerszintű modellek a tervezési részben.

**Állapot:** **Nem teljesül.**

A 4. fejezet (Szoftveralkalmazás tervezése) tartalmaz „Adatfolyam diagram” alcímet, és a szöveg **leírja** az adatfolyamot, de a dokumentumban **nincs beszúrt diagram** (sem UML, sem adatfolyam, sem más formális modell). A követelmény a **modellek alkalmazását** (tehát a diagram megjelenítését) kéri.

**Javaslat:** Készíts (pl. Draw.io, PlantUML, vagy más eszközzel) legalább egy adatfolyam- vagy komponensdiagramot (esetleg egyszerű UML részt), mentsd képként (vagy PDF-ként), és illeszd be a 4.1.3 szekcióba `figure` környezetben, címkével és sorszámmal, majd a szövegben hivatkozz rá.

---

## Összefoglaló táblázat

| # | Követelmény                          | Állapot        | Teendő |
|---|--------------------------------------|----------------|--------|
| 1 | Tárgyilagos, személytelen megfogalmazás | Nem teljesül   | Szöveg átfogalmazása (passzív / tárgyilagos) |
| 2 | Szakszerű terminológia               | Megfelel       | – |
| 3 | Rövidítések + jegyzék                 | Részben        | Rövidítésjegyzék, ha kéri a Tanszék |
| 4 | IEEE hivatkozási stílus               | Nem teljesül   | biblatex style=ieee, szövegközi \cite |
| 5 | Helyesírás                           | Nem ellenőrizve | Külön átnézés |
| 6 | Fejezetszerkezet                     | Megfelel       | – |
| 7 | Magyar kivonat + angol Abstract      | Nem teljesül   | Kivonat és Abstract (max 1–1 oldal) |
| 8 | Times 12 pt, 1,5 sorköz, sorkizárt   | Részben        | \onehalfspacing beállítása |
| 9 | Margók (bal 3, többi 2,5 cm)         | Nem teljesül   | geometry: left=3cm |
| 10| Ábrák/táblázatok sorszám + hivatkozás | Nem teljesül   | Figure/table + \ref a szövegben |
| 11| Forráskód monospace                   | Részben        | \texttt OK; hosszabb kód: listings |
| 12| UML / rendszermodellek               | Nem teljesül   | Adatfolyam (vagy UML) diagram beszúrása |

---

**Összegzés:** A szerkezet és a terminológia nagyjából rendben van; a legtöbb hiány a **formai** (margó, sorköz), a **kivonat/Abstract**, az **irodalom (IEEE + hivatkozások)** és a **ábrák/diagramok** bevezetésénél van. A tárgyilagos stílusra érdemes a teljes szöveget átírni.
