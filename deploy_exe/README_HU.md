# Másik gépen futtatás – PyInstaller (gyakorlatban hogyan néz ki)

## 1. Kell-e külön venv a célgépre?

**Nem.** A PyInstaller egy **önálló mappát** (onedir) vagy egy nagy **.exe**-t készít, amibe **belecsomagolja** a Python futtatókörnyezetét és a függőségeket. A végfelhasználónak **nem kell** Python vagy `pip install`.

A **tiszta venv** csak **a te gépeden** kell a *buildhez* (hogy ne legyen szemét a csomagban).

## 2. Egy darab .exe vs. „mappa + exe”

| Megoldás | Előny | Hátrány |
|----------|--------|---------|
| **`--onedir` (mappa + exe)** | Gyors indulás, stabil, TF+Torch+OpenCV így szokás | Több fájl (zipben átadod az egész mappát) |
| **`--onefile` (egy .exe)** | Egy fájl | Nagy méret, **lassú első indulás** (kicsomagolás temp-be), vírusirtók gyakrabban zavarják |

**Javaslat:** először **`onedir`**. Ha kell, később próbálhattok onefile-t.

## 3. A modellek hol legyenek?

A nagy fájlokat (`.pth`, `.keras`) **nem kötelező** az exe „belsejébe” tenni (lassú build, óriási exe).

**Gyakorlatban:**

1. Build után a **`dist\LiveCapture\`** (vagy hasonló) mappába **másolod** ugyanazt a struktúrát, mint a repóban:
   - `checkpoints\`
   - `checkpoints_unet\`
   - `classificator_models\`
2. A `live_capture_app.py` **PyInstaller alatt** a `_BASE_DIR`-t az **.exe mappájára** állítja → ezek a mappák **az .exe mellett** legyenek.

Így a **frissített modell** = fájlcsere, **újra build nélkül**.

## 4. Mit visz át a másik laptopra?

- A **teljes `dist\...\` mappát** (zip / pendrive), **plusz** a fenti modellek mappái, ha nem voltak a buildben.
- Opcionális: **Microsoft Visual C++ Redistributable** (x64), ha OpenCV/Torch miatt hiányzik DLL.

## 5. Python verzió – ugyanaz legyen, mint a zenbook venvben

A PyInstaller **abba a Pythonba** csomagolja be a futtatókörnyezetet, **amivel a build fut**. Ha mással buildelsz, mint amivel fejlesztesz, **más DLL-ek / wheel-ek** kerülhetnek az exe-be → furcsa összeomlások.

**Ebben a repóban** (ellenőrizve a `zenbook_venv\pyvenv.cfg` alapján):

| Környezet | Python |
|-----------|--------|
| **zenbook_venv** | **3.11.5** (alap: `C:\Program Files\Python311`) |
| **build_venv** (ha használod) | szintén a zenbook venv Pythonjával lett létrehozva → **3.11.x** |

**Build előtt mindig ellenőrizd** (aktivált venv mellett):

```bat
where python
python --version
python -c "import sys; print(sys.executable)"
```

A kimenetnek a **zenbook_venv\Scripts\python.exe**-re kell mutatnia, és **3.11.x**-nek lennie.

Ha a Cursor / más terminál **másik** `python.exe`-t használ (pl. 3.10 vagy 3.12), **ne azzal** futtasd a `build_exe.bat`-ot – előbb `zenbook_venv\Scripts\activate`.

A `requirements_app.txt` **nem** köti le a Python patch szintet; a **build gép 3.11.x** maradjon konzisztens a fejlesztéssel.

## 5b. Új gép – dev telepítő (exe nélkül)

A repó gyökerében: **`install.bat`** vagy

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_dev.ps1
```

- **RepoUrl üres:** a már meglévő klónozott repóban hozza létre a `zenbook_venv`-et és telepíti a függőségeket.
- **Saját GitHub később:** add meg a linket, pl.  
  `.\scripts\install_dev.ps1 -RepoUrl 'https://github.com/Felhasznalo/repo'`  
  (üresen hagyható a script paramétere, amíg nincs repo – akkor helyi módot használj.) **Nincs Git?** A script `winget`-tel megpróbálja telepíteni; vagy `-UseZip`.
- **PyInstaller is kell:** `-IncludeBuildDeps`
- **Python:** hiány esetén `winget` vagy python.org csendes telepítő; `-SkipPythonInstall` / `-PythonMinor 12` ha kell.
- **PyTorch:** alapból **CPU** wheel (új, driver nélküli Windowsra megbízható). **NVIDIA GPU:** `install.bat -GpuTorch` (CUDA 12.4 wheel; kell megfelelő driver).
- **VC++ Redistributable x64:** `winget`-tel megpróbálja (OpenCV DLL-ek); `-SkipVCRedist` kikapcsolja.
- **TensorFlow** gond esetén: `-SkipTensorFlow` (az app így is futhat, TF nélkül).
- **Modellek:** a súlyok **nincsenek** a Git repóban (méret); másold őket külön — [`MODELLOK_MASOLASA.txt`](MODELLOK_MASOLASA.txt).

## 6. Build lépések (fejlesztői gép)

1. **zenbook_venv aktiválása**, majd `pip install -r requirements_app.txt` (+ opcionálisan `pygrabber` kameralistához).
2. `pip install pyinstaller` (vagy `pip install -r deploy_exe\requirements_build.txt`).
3. A repó **gyökeréből** `build_exe.bat` / `build_exe_console.bat` (vagy `deploy_exe\build_onedir.ps1`).
4. Másold a modellek mappáit a `dist\...` mellé (lásd `MODELLOK_MASOLASA.txt`).

## 7. Mi van ebben a mappában?

- `README_HU.md` – ez a leírás  
- `build_onedir.ps1` – PyInstaller indítás a repó gyökeréből  
- `live_capture_app.spec` – beállítások (hiddenimports, stb. finomítható)  
- `MODELLOK_MASOLASA.txt` – pontos másolandó lista  

Ha a build hibázik (főleg TensorFlow), a spec-ben bővíthető a `hiddenimports` / `collect_submodules` – erről a PyInstaller + TF dokumentáció ad részleteket.

## 8. Az exe „nem indul”, vagy az Explorer furcsán viselkedik

- **Ablak nélküli exe (`console=False`)** induláskor összeomlhat **úgy, hogy nem látszik hiba**. Ilyenkor:
  - futtasd a **konzolos debug buildet**: a repó gyökerében `build_exe_console.bat` → kimenet: `dist\LiveCapture_console\`
  - vagy: `set LIVECAPTURE_CONSOLE=1` majd `pyinstaller --clean --noconfirm deploy_exe\live_capture_app.spec`
  - nézd meg a **`LiveCapture_crash.log`** fájlt: az **exe mellett**, és ha oda nem írható, a **`%TEMP%\LiveCapture_crash.log`** helyen is (a `live_capture_bootstrap.py` belépő mindkettőbe ír).
- **Nagy `dist\LiveCapture` mappa** (TensorFlow + PyTorch + sok DLL): a **Fájlkezelő** lassulhat vagy instabil lehet, különösen **OneDrive** alatt vagy **előnézettel**.
  - tedd a buildet **nem szinkronizált** mappába (pl. `C:\build\LiveCapture\`), vagy kapcsold ki az előnézetet;
  - ideiglenesen **kizárhatod** a mappát a vírusirtó vizsgálatából (csak saját gépen, óvatosan).
