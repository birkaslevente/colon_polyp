# Hordozható csomag: exe + modellek egy zipben. Célgépen nincs GitHub és Python.
# Csak zenbook_venv Python 3.11. Inno Setup varázsló csak ha az ISCC.exe megvan.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Py = Join-Path $RepoRoot "zenbook_venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "Megall: nincs zenbook_venv. Nem keszul csomag." -ForegroundColor Red
    exit 1
}
$Ver = (& $Py -c "import sys; print('%d.%d' % (sys.version_info[0], sys.version_info[1]))").Trim()
if ($Ver -ne "3.11") {
    Write-Host "Megall: Python $Ver, 3.11 kell. Nem keszul csomag." -ForegroundColor Red
    exit 1
}

# A dist a OneDrive-on kivul keszul: a mappa torlese kulonben WinError 5-ot ad.
$Stage = Join-Path $env:LOCALAPPDATA "LiveCapture-build\dist"
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$Spec = Join-Path $RepoRoot "deploy_exe\live_capture_app.spec"
& $Py -m PyInstaller --noconfirm --distpath $Stage $Spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Dist = Join-Path $Stage "LiveCapture"
if (-not (Test-Path -LiteralPath (Join-Path $Dist "LiveCapture.exe"))) {
    Write-Host "Megall: nincs dist\LiveCapture\LiveCapture.exe" -ForegroundColor Red
    exit 1
}

$ModelDirs = @("checkpoints_0409", "checkpoints_unet", "classificator_models")
foreach ($Dir in $ModelDirs) {
    $Src = Join-Path $RepoRoot $Dir
    if (-not (Test-Path -LiteralPath $Src)) {
        Write-Host "Megall: hianyzik a modellmappa: $Src" -ForegroundColor Red
        exit 1
    }
    $Dest = Join-Path $Dist $Dir
    $PrevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & robocopy $Src $Dest /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    $CopyCode = $LASTEXITCODE
    $ErrorActionPreference = $PrevEap
    if ($CopyCode -ge 8) {
        Write-Host "Megall: masolas sikertelen: $Dir (robocopy $CopyCode)" -ForegroundColor Red
        exit 1
    }
}

$Inditas = @"
LiveCapture

1. Csomagold ki ezt a mappat egy tetszoleges helyre (pendrive vagy OneDrive is jo).
2. Inditsd a LiveCapture.exe fajlt. Python es GitHub nem kell.
3. Ha az inditas hianyzo DLL-t ir (VCRUNTIME140 vagy hasonlo), telepitsd a
   Microsoft Visual C++ 2015-2022 x64 Redistributable csomagot, majd inditsd ujra.

A modellmappak az exe mellett vannak: checkpoints_0409, checkpoints_unet, classificator_models.
"@
Set-Content -LiteralPath (Join-Path $Dist "INDITAS.txt") -Value $Inditas -Encoding UTF8

$Zip = Join-Path $env:LOCALAPPDATA "LiveCapture-build\LiveCapture-portable.zip"
if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}
& tar.exe -a -c -f $Zip -C $Stage "LiveCapture"
if ($LASTEXITCODE -ne 0) {
    Write-Host "A zip nem keszult el. A masolhato mappa kesz: $Dist" -ForegroundColor Yellow
    exit 1
}
Write-Host "Zip: $Zip" -ForegroundColor Green
$OutDir = Join-Path $RepoRoot "dist"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -LiteralPath $Zip -Destination (Join-Path $OutDir "LiveCapture-portable.zip") -Force

$IsccCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($Iscc) {
    & $Iscc "/DRepoDist=$Dist" (Join-Path $RepoRoot "installer\LiveCapture.iss")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Inno telepito: dist\LiveCapture-setup.exe" -ForegroundColor Green
} else {
    Write-Host "ISCC.exe nincs. A zip kesz, varazslo nem keszul. A masolashoz a zip eleg." -ForegroundColor Yellow
}
