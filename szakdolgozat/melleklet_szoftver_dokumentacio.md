# Melléklet X – A kolorektális polip szegmentáló modell szoftveres fejlesztésének dokumentációja

## 1. Cél és kontextus

### 1.1 A dokumentáció célja
Jelen melléklet a diplomamunkában bemutatott kolorektális polip szegmentáló rendszer (DeepLabV3+ architektúra) szoftveres megvalósítását és fejlesztési lépéseit dokumentálja. Célja, hogy bemutassa a `colon_short_szd.ipynb` Jupyter Notebook felépítését szoftvermérnöki és adatkutatói (Data Science/ML) szemszögből, biztosítva a kutatás átláthatóságát és a kód reprodukálhatóságát.

### 1.2 Kapcsolat a dolgozattal
Míg a dolgozat 3. fejezete (Módszertan) a modell elméleti hátterét, az architektúra választásának indoklását és a képfeldolgozási elveket tárgyalja, ez a melléklet szigorúan az implementációra fókuszál: hogyan történt a PyTorch osztályok példányosítása, az adathalmazok betöltése, valamint a betanítás és a kiértékelés szoftveres megvalósítása.

### 1.3 Felhasználási szcenárió
A szoftver elsődlegesen offline modelltanításra és validációra készült (Jupyter környezetben). A kimenetként kapott betanított modell súlyok (`.pth` fájl) a későbbiekben felhasználhatók önálló inferencia szkriptekben (pl. `live_capture_app.py`), valós idejű vagy rögzített endoszkópos videók/képek feldolgozására.

---

## 2. Fejlesztési környezet és függőségek

A projekt fejlesztése és futtatása lokális, illetve felhőalapú GPU környezetben történt a megfelelő teljesítmény elérése érdekében.

### 2.1 Hardverkörnyezet
- **Számítási egység:** Nvidia GPU (pl. RTX széria vagy Google Colab T4) a gyorsított tenzorműveletekhez. Egy epoch tanítási ideje a batch mérettől és a GPU teljesítményétől függően változik.
- **Rendszermemória:** Minimum 16 GB RAM javasolt a képek betöltéséhez és az adat-augmentációhoz.

### 2.2 Szoftverkörnyezet és verziók
A fejlesztés Python nyelven, Jupyter Notebook / Cursor / VSCode környezetben zajlott. A legkritikusabb függőségek a következők:
- **Operációs rendszer:** Windows / Linux (WSL)
- **Python verzió:** 3.10+
- **PyTorch:** `torch` és `torchvision` (CUDA támogatással)
- **Egyéb elengedhetetlen könyvtárak:** `numpy`, `matplotlib`, `Pillow`, `scikit-learn`

### 2.3 Környezet felállítása
A projekt egy dedikált virtuális környezetben (`venv`) futtatható. A telepítés a következőképpen történik:
```bash
python -m venv venv
# Windows esetén:
venv\Scripts\activate
# Linux/Mac esetén:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Rendszerarchitektúra magas szinten

A rendszer adatfolyama moduláris felépítésű, követve a gépi tanulási projektek klasszikus életciklusát:
1. **Adatbetöltés és -felosztás:** Képek és maszkok szinkronizált betöltése, majd felosztása (train/val/test).
2. **Adat- és maszkolás előfeldolgozás:** Képek transzformációja (augmentáció) Pytorch Dataset osztályok segítségével.
3. **Modell definíció:** A `torchvision` könyvtár DeepLabV3+ modelljének inicializálása MobileNetV2 encoderrel.
4. **Tanítási ciklus:** Mini-batch alapú iteráció, veszteségfüggvény számítása, backpropagation.
5. **Kiértékelés és mentés:** Validációs metrikák (IoU, Dice) számítása, a legjobb modellállapot (checkpoint) automatikus mentése.

*(Az adatfolyam vizuális ábrázolását a dolgozat vonatkozó fejezete tartalmazza.)*

---

## 4. Adatkezelés és előfeldolgozás

### 4.1 Címtárszerkezet
A kód az alábbi dedikált mappastruktúrát várja az adatbázisokhoz (Kvasir-SEG és a SZE Dataset beolvasztása után):

```text
projekt_gyökér/
├── images/
│   ├── polyp_1.jpg
│   ├── polyp_2.jpg
│   └── ...
├── masks/
│   ├── polyp_1.jpg (vagy .png)
│   ├── polyp_2.jpg
│   └── ...
└── checkpoints/
```
A betöltő szkript párosítja az azonos nevű képeket és maszkokat.

### 4.2 PyTorch Dataset és Dataloader
A képek beolvasása egy egyedi, `torch.utils.data.Dataset` osztályból származtatott implementáción keresztül történik. 
- A **DataLoader** felel a képek kötegeléséért (batching), jellemzően `batch_size=4` értékkel, a GPU memória korlátai miatt.
- A betanító DataLoader esetében a `shuffle=True` beállítás biztosítja az adatok véletlenszerűsítését.

### 4.3 Transzformációk (Augmentáció)
Az augmentáció a `torchvision.transforms` segítségével valósul meg a betanítási fázisban:
- **RandomScale / Resize:** A képek méretének véletlenszerű változtatása.
- **RandomCrop (513x513):** A hálózat rögzített bemeneti méretének biztosítása.
- **RandomHorizontalFlip / RandomRotation:** Forgatások és tükrözések a modell robusztusságának növelésére.
- **Normalize:** Az ImageNet statisztikáinak (mean, std) megfelelő normalizáció, amely kritikus a transzfer tanulás (pretrained súlyok) sikerességéhez.

---

## 5. Modellkomponens (DeepLabV3+)

A rendszer központi eleme a szegmentáló hálózat, amelynek inicializálása a `torchvision` modulból történik.

### 5.1 Inicializálás és architektúra
A kód a `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` (vagy a korábbi V2-es alapú) implementációt hívja meg.
- **Pretrained súlyok:** `weights='DEFAULT'` (COCO/VOC alapú előtanítás).
- **Kimeneti réteg (Classifier):** Mivel a feladat bináris szegmentálás (háttér vs. polip), az utolsó konvolúciós réteget a kód lecseréli, hogy pontosan `num_classes=2` kimeneti csatornája legyen.

### 5.2 Veszteségfüggvény és Optimalizáló
- **Veszteségfüggvény (Loss):** `nn.CrossEntropyLoss()`, amely pixel-szintű többosztályos (itt 2 osztályos) klasszifikációra optimalizál.
- **Optimalizáló algoritmus:** `torch.optim.Adam` inicializálva `lr=0.001` (1e-3) tanulási rátával.
- **LR Scheduler:** `ReduceLROnPlateau` osztály, amely automatikusan csökkenti a tanulási rátát (pl. `factor=0.1`), ha a validációs IoU nem javul meghatározott számú (patience) epoch után.

---

## 6. Tanítási és validációs folyamat

A tanítás logikája egy kifejezett tréning ciklusba van szervezve.

### 6.1 Tanító ciklus (Train Epoch)
Minden epochban a kód végigiterál a betanító adathalmazon:
1. Az adatok GPU-ra másolása (`.to(device)`).
2. Előrehaladás (`forward pass`): `outputs = model(images)`.
3. Veszteség számítása a maszkokhoz viszonyítva.
4. Visszaterjesztés (`backward pass`): `loss.backward()`.
5. Súlyok frissítése: `optimizer.step()`, és a gradiensek nullázása: `optimizer.zero_grad()`.

### 6.2 Validáció és Early Stopping
Minden tanítási epoch végén a hálózat `model.eval()` módba kapcsol, és lefut a validációs adathalmazon.
- **Metrikák:** Dice együttható és Intersection over Union (IoU) számítása történik pixel-szinten.
- **Early Stopping:** A kód figyeli a validációs IoU-t. Ha bizonyos számú epochon keresztül (pl. `patience=7`) nincs javulás, a tanítás korábban leáll, megelőzve a túltanulást (overfitting).
- **Naplózás (Logging):** A metrikák (loss, lr, IoU) historikus adatai memóriában (szótárként) tárolódnak, majd a futás végén kirajzolódnak a matplotlib segítségével.

---

## 7. Eredmények reprodukálhatósága

A gépi tanulási modellek megbízhatóságának alapja a reprodukálhatóság. Ennek érdekében a kód az alábbi elveket követi:

### 7.1 Seed beállítások
A futtatás elején egy központi függvény beállítja a determinisztikus generátorokat:
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNN backend determinizmus beállítása (opcionális, de ajánlott)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### 7.2 Hiperparaméter konfiguráció
A végső betanított modellhez tartozó főbb hiperparaméterek (melyekkel az eredmények reprodukálhatók):
- **Batch size:** 4
- **Max Epochs:** 50 (Early stoppinggal megállhat hamarabb is, pl. 15-20 körül)
- **Kezdeti LR:** 0.001
- **Optimizer:** Adam
- **Kép bemeneti méret:** 513x513

---

## 8. Kimenetek és használat (Inferencia)

### 8.1 Checkpoint-kezelés
A kód csak a legjobb modellt (best IoU) menti le a lemeze: `checkpoints/best_model_epoch_{epoch}.pth`. Ez a fájl tartalmazza a modell súlyait (`state_dict`), az optimalizáló állapotát és a tanulási metrikákat.

### 8.2 Predikciós Pipeline kódminta
A betanított modell gyakorlati felhasználása (inference) az alábbi minimális kódrészlettel végezhető el egy új képre:

```python
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image
import torchvision.transforms as T

# 1. Modell inicializálása és súlyok betöltése
model = deeplabv3_mobilenet_v3_large(num_classes=2)
checkpoint = torch.load('checkpoints/best_model_epoch_12.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Kép betöltése és előfeldolgozása (ugyanaz a normalizáció, mint tanításnál)
img = Image.open('test_image.jpg').convert('RGB')
transform = T.Compose([
    T.Resize((513, 513)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0) # Batch dimenzió hozzáadása

# 3. Predikció (Inferencia)
with torch.no_grad():
    output = model(input_tensor)['out']
    # Osztály valószínűségek és argmax (0: háttér, 1: polip)
    prediction = torch.argmax(output.squeeze(), dim=0).byte().cpu().numpy()

# A 'prediction' változó most egy 513x513-as bináris maszkot tartalmaz.
```

---

## 9. Ismert limitációk és jövőbeli bővítés

Bár a jelenlegi szoftverarchitektúra stabilan működik, szoftvermérnöki szempontból az alábbi korlátokkal és fejlesztési lehetőségekkel érdemes számolni:
- **Korlátok:** A `batch_size=4` relatív kicsi, amely a GPU memória (VRAM) korlátaiból fakad. Ez némileg zajosabb gradiens-frissítést eredményez. Az adathalmaz mérete (bár kiterjesztett) még mindig hagy teret a túltanulásnak, amit az early stopping hivatott tompítani.
- **Továbbfejlesztés (Architektúra szinten):** Konfigurációs fájlok (pl. `config.yaml`) bevezetése a notebookba keménykódolt változók (hardcoded variables) helyett.
- **Továbbfejlesztés (Inferencia):** A valós idejű endoszkópos felhasználáshoz a betanított PyTorch modell ONNX vagy TensorRT formátumba történő konvertálása javasolt a másodpercenkénti képkockaszám (FPS) drasztikus növelése érdekében.
