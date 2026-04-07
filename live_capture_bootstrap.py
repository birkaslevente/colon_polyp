# -*- coding: utf-8 -*-
"""
PyInstaller belépő: csak stdlib, még a nehéz importok előtt naplóz.
Így import/DLL hiba esetén is marad LiveCapture_crash.log + konzol kimenet.
Forrásból: python live_capture_bootstrap.py
"""
import faulthandler
import os
import sys
import traceback


def _exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _log_paths() -> tuple[str, str]:
    primary = os.path.join(_exe_dir(), "LiveCapture_crash.log")
    alt = os.path.join(os.environ.get("TEMP", os.getcwd()), "LiveCapture_crash.log")
    return primary, alt


def _emit(msg):
    line = msg.rstrip("\n")
    print(line, flush=True)
    for path in _log_paths():
        try:
            with open(path, "a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            continue


def main():
    primary, alt = _log_paths()
    # új futás: töröljük / új fejléc (mindkét helyen, ha írható)
    for path in (primary, alt):
        try:
            with open(path, "w", encoding="utf-8", errors="replace") as f:
                f.write(f"=== LiveCapture bootstrap pid={os.getpid()} ===\n")
                f.write(f"exe_dir={_exe_dir()}\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass

    try:
        fh = open(primary, "a", encoding="utf-8", errors="replace")
        faulthandler.enable(file=fh)
    except OSError:
        try:
            fh = open(alt, "a", encoding="utf-8", errors="replace")
            faulthandler.enable(file=fh)
        except OSError:
            faulthandler.enable()

    _emit(f"log: {primary} (masolat: {alt})")

    try:
        _emit("import live_capture_app …")
        import live_capture_app  # noqa: PLC0415

        _emit("import OK, run_app() …")
        live_capture_app.run_app()
        _emit("run_app() vége (normál kilépés)")
    except BaseException:
        tb = traceback.format_exc()
        _emit("FATAL (bootstrap elkapva):\n" + tb)
        try:
            input("Nyomj Entert a bezáráshoz… ")
        except Exception:
            pass
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
