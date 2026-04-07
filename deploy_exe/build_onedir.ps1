# PyInstaller onedir build – a repó GYOKEREBOL futtasd: .\deploy_exe\build_onedir.ps1
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Telepitsd: pip install pyinstaller" -ForegroundColor Red
    exit 1
}

Write-Host "Build environment: ugyanabban a Python/venv-ben futtasd, ahol a TensorFlow telepitve van." -ForegroundColor Cyan
Write-Host "Ha TensorFlow DLL hiba van futasnal, telepitsd a Microsoft Visual C++ 2015-2022 x64 Redistributable csomagot." -ForegroundColor Yellow

$spec = Join-Path $RepoRoot "deploy_exe\live_capture_app.spec"
pyinstaller --clean --noconfirm $spec

Write-Host ""
Write-Host "Kesz. Nezd meg: $RepoRoot\dist\LiveCapture\" -ForegroundColor Green
Write-Host "Masold be a modelleket – lasd deploy_exe\MODELLOK_MASOLASA.txt" -ForegroundColor Yellow
