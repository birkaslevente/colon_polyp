# Terv: szövegek az „Ide kell még szöveg” helyekre

A következő négy bekezdés vázlat a megvalósított alkalmazás (élő kép + gombnyomásra szegmentálás és klasszifikáció) és a dolgozat többi stílusa alapján készült. A [X], [Y], [Z] helyőrzőket később konkrét mérésekkel helyettesítsd.

---

## 1.2.1 – Élő kép megjelenítés és igény szerinti (gombnyomásra) elemzés követelményei

**Tervezet:**

A klinikai környezetben a rendszer két elkülönülő követelményt teljesít: az élő videó folyamatos, zökkenőmentes megjelenítését és a felhasználó által kezdeményezett, azonnali elemzést. Az élő megjelenítésnél a megjelenítési sebességnek (FPS) el kell érnie a vizsgálat során kényelmes érzetet nyújtó szintet (általában 25–30 képkocka/másodperc), hogy az orvos ne érezze a késleltetést vagy a megakadást. Az elemzés nem minden képkockán történik, hanem gombnyomásra (vagy rövidített billentyűre): ekkor a rendszer a pillanatnyi képkockát rögzíti, szegmentálja és osztályozza. Ennél a működésmódnál a fő teljesítménymutató a válaszidő (end-to-end latency), azaz a gombnyomás és a képernyőn megjelenő szegmentált eredmény és klasszifikáció közötti idő, amelynek célszerűen 100 milliszekundum alatt kell maradnia az észrevehetetlen támogatás érdekében.

---

## 1.3.2 – Moduláris Python alkalmazás élő videó megjelenítésre és gombnyomásra történő szegmentálásra és klasszifikációra

**Tervezet:**

Második célkitűzésem egy moduláris Python alkalmazás megvalósítása, amely élő videóforrásból (kamera, videófájl vagy szimulációs képmappa) folyamatosan megjeleníti a képet, és a felhasználó gombnyomására („Capture \& Analyze”) a pillanatnyi képkockán elvégzi a polip szegmentálást és a klasszifikációt. A szoftver „hídként” funkcionál a bemeneti eszköz és a mélytanulási modellek között: a videófolyamot dedikált szál kezeli, míg az elemzés igény szerint, a rögzített képkockán fut. A megjelenítés három panelből áll: élő kép, a hálózatnak beadott (középre vágott) bemenet és a szegmentált polip eredmény; a klasszifikátor kimenete (pl. adenóma/hiperplasztikus) szintén megjelenik. Az architektúra lehetővé teszi külső osztályozó modulok cseréjét a meglévő interfész révén.

---

## 5.2.1 – Megjelenítési sebesség (FPS) és igény szerinti elemzési idő

**Tervezet:**

A rendszer teljesítményét két szempont szerint mértem. (1) A megjelenítési sebesség (FPS) azt mutatja, hogy az élő videó hány képkocka/másodperc ütemben frissül a felületen; ezt a capture és a megjelenítő szál együttes kapacitása határozza meg. Célom volt, hogy ez a érték stabilan maradjon a [Z] FPS körül vagy felett, így az orvos zökkenőmentes élő képet lásson. (2) Az igény szerinti elemzési idő (egy elemzés menetideje) azt az időintervallumot jelenti, amely a gombnyomás (vagy Space) és a szegmentált kép és a klasszifikációs eredmény megjelenése között eltelik. Ezt külön mérésekkel (időbélyegek vagy nagy felbontású időmérés) határoztam meg; az eredmények szerint egy elemzés [X] ms körüli időt vesz igénybe, ami a 100 ms-os célérték alatt marad, így a válasz valós idejűnek érződik a felhasználó számára.

---

## 5.3.1 – A gombnyomásra történő elemzés korlátai

**Tervezet:**

Az eredmények elemzése során a gombnyomásra történő elemzés működési módjának sajátos korlátai váltak nyilvánvalóvá. Mivel a szegmentálás és a klasszifikáció nem minden képkockán, hanem csak a felhasználó által kiválasztott pillanatban fut, a rendszer nem nyújt folyamatos, képkockánkénti figyelmeztetést; a polip detektálása tehát az orvos döntésén múlik, mikor nyomja meg a gombot. Emellett a nagyfelbontású bemenet előfeldolgozása (középre vágás, átméretezés) és a CPU–GPU adatátvitel továbbra is jelentős késleltetési tényező maradhat, különösen gyengébb hardveren. A Python interpreter és a GIL hatása egyetlen, igény szerinti inferencia esetén kevésbé kritikus, mint folyamatos feldolgozásnál, de a GUI szál és az inferencia szinkronizálása további optimalizációs lehetőséget jelent a jövőben.

---

## Használat

- Másold a megfelelő bekezdést a `szakdolgozat.tex`-be az „Ide kell még szöveg.” helyére.
- Cseréld ki a [X], [Y], [Z] helyőrzőket a tényleges méréseidre (ms, FPS, százalék).
- Szükség szerint bővítsd irodalmi hivatkozással vagy részletesebb mérési leírással.
