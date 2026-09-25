# PyInstaller onedir build – a repó GYOKEREBOL futtasd: .\deploy_exe\build_onedir.ps1
# Csak a zenbook_venv Python 3.11-gyel. A modelleket a build_installer.ps1 másolja.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Py = Join-Path $RepoRoot "zenbook_venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "Megall: a build csak a zenbook_venv Pythonjaval futhat." -ForegroundColor Red
    exit 1
}
$Ver = (& $Py -c "import sys; print('%d.%d' % (sys.version_info[0], sys.version_info[1]))").Trim()
if ($Ver -ne "3.11") {
    Write-Host "Megall: Python $Ver, 3.11 kell. Nem keszul csomag." -ForegroundColor Red
    exit 1
}

Write-Host "Build environment: ugyanabban a Python/venv-ben futtasd, ahol a TensorFlow telepitve van." -ForegroundColor Cyan
Write-Host "Ha TensorFlow DLL hiba van futasnal, telepitsd a Microsoft Visual C++ 2015-2022 x64 Redistributable csomagot." -ForegroundColor Yellow

$spec = Join-Path $RepoRoot "deploy_exe\live_capture_app.spec"
& $Py -m PyInstaller --noconfirm $spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Kesz. Nezd meg: $RepoRoot\dist\LiveCapture\" -ForegroundColor Green
Write-Host "A hordozhato ziphez: .\deploy_exe\build_installer.ps1" -ForegroundColor Yellow
