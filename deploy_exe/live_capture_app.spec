# -*- mode: python ; coding: utf-8 -*-
# Futtatas: repó gyökeréből deploy_exe\build_onedir.ps1
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

REPO = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

# Konzolos debug build: cmd-ben: set LIVECAPTURE_CONSOLE=1 majd pyinstaller ...
# Így látszik a traceback; a kimenet: dist\LiveCapture_console\
_DEBUG_CONSOLE = os.environ.get("LIVECAPTURE_CONSOLE", "").strip() in ("1", "true", "yes", "on")
_APP_NAME = "LiveCapture_console" if _DEBUG_CONSOLE else "LiveCapture"

block_cipher = None

# Opcionális: pygrabber a kamera-nevek listájához – ha nincs telepítve, vedd ki a hiddenimports sorból
hiddenimports = [
    "live_capture_app",
    "customtkinter",
    "network",
    "network.modeling",
    "utils.ext_transforms",
    "segmentation_models_pytorch",
    "cv2",
    "PIL._tkinter_finder",
]

binaries = []
datas = []

# TensorFlow packaging (klasszifikátorhoz):
# - hiddenimports: modulok rekurzívan
# - binaries: natív DLL-ek (pl. _pywrap_tensorflow_internal függőségei)
# - datas: runtime adatfájlok
try:
    hiddenimports += collect_submodules("tensorflow")
    binaries += collect_dynamic_libs("tensorflow")
    datas += collect_data_files("tensorflow")
except Exception as e:
    print(f"[spec] TensorFlow collect figyelmeztetés: {e}")

# Keras külön csomagként is jelen lehet egyes környezetekben
try:
    hiddenimports += collect_submodules("keras")
    datas += collect_data_files("keras")
except Exception as e:
    print(f"[spec] Keras collect figyelmeztetés: {e}")

# CustomTkinter: JSON témák / assetek — nélkülük az exe néma összeomlás lehet
try:
    hiddenimports += collect_submodules("customtkinter")
    datas += collect_data_files("customtkinter")
except Exception as e:
    print(f"[spec] customtkinter collect figyelmeztetés: {e}")

a = Analysis(
    [os.path.join(REPO, "live_capture_bootstrap.py")],
    pathex=[REPO],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=_APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_DEBUG_CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=_APP_NAME,
)
