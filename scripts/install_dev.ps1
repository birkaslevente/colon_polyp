#Requires -Version 5.1
<#
.SYNOPSIS
    Dev kornyezet: venv + pip install (exe nelkul).

.DESCRIPTION
    - RepoUrl ures: a script melletti repogyokerben (scripts\..) vagy a jelenlegi mappaban
      keresi a requirements_app.txt-t, es ott hozza letre a venv-et.
    - RepoUrl megadva: git clone vagy GitHub ZIP letoltes, majd ugyanaz.

    A GIT linket IDE ILLD BE, ha kesz a sajat repo (vagy futtasd uresen, ha mar klónoztál).

.PARAMETER RepoUrl
    Pl. https://github.com/Felhasznalo/DeepLabV3Plus-Pytorch  (ures = csak helyi repó)

.PARAMETER Branch
    Alap: main (master helyett allitsd at, ha a forkod mast hasznal)

.PARAMETER UseZip
    Git helyett ZIP letoltes (GitHub archive URL)

.PARAMETER TargetDir
    Uj klónozáshoz / kicsomagoláshoz celkonyvtar (ures = .\DeepLabV3Plus-Pytorch a jelenlegi mappa alatt)

.PARAMETER VenvName
    Virtualenv mappa neve a repó gyökerében (alap: zenbook_venv)

.PARAMETER IncludeBuildDeps
    PyInstaller + build requirements (deploy_exe\requirements_build.txt)

.PARAMETER SkipPythonInstall
    Ha nincs Python: ne probaljon telepiteni (hibaval leall). Alap: hiany eseten winget, majd python.org csendes telepito.

.PARAMETER PythonMinor
    Telepitendo / preferalt 3.x alverzio (10, 11 vagy 12). Alap: 11.

.PARAMETER GpuTorch
    PyTorch + torchvision CUDA 12.4 wheel (NVIDIA driver szukseges). Alap: CPU wheel — tiszta Windowson megbizhato.

.PARAMETER SkipGitInstall
    Ne telepitsen Git-et winget-tel, ha hianyzik (clone elott).

.PARAMETER SkipVCRedist
    Ne probalja a VC++ 2015-2022 x64 redist telepitest (OpenCV DLL-ekhez ajanlott uj Windowson).

.PARAMETER SkipTensorFlow
    Ne telepitsen tensorflow csomagot (az app akkor is fut, ha TF import opcionalis).
#>
param(
    [string]$RepoUrl = "",
    [string]$Branch = "main",
    [switch]$UseZip,
    [string]$TargetDir = "",
    [string]$VenvName = "zenbook_venv",
    [switch]$IncludeBuildDeps,
    [switch]$SkipPythonInstall,
    [ValidateSet(10, 11, 12)]
    [int]$PythonMinor = 11,
    [switch]$GpuTorch,
    [switch]$SkipGitInstall,
    [switch]$SkipVCRedist,
    [switch]$SkipTensorFlow
)

$ErrorActionPreference = "Stop"
# Régi Windows / alapértelmezett TLS miatt (GitHub ZIP, python.org)
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
}
catch { }

function Test-RepoRoot {
    param([string]$Path)
    return (Test-Path -LiteralPath (Join-Path $Path "requirements_app.txt"))
}

function Get-GitHubZipUrl {
    param([string]$Url, [string]$Br)
    $u = $Url.Trim().TrimEnd("/")
    if ($u.EndsWith(".git")) { $u = $u.Substring(0, $u.Length - 4) }
    if ($u -notmatch "github\.com[:/]([^/]+)/([^/]+)$") {
        throw "RepoUrl nem GitHub forma. Hasznalj https://github.com/Tulajdonos/Repo alakot."
    }
    $owner = $Matches[1]
    $repo = $Matches[2]
    return "https://github.com/$owner/$repo/archive/refs/heads/$Br.zip"
}

function Update-SessionPathFromMachineAndUser {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ";"
}

function Test-PythonVersionLine {
    param([string]$Line)
    return $Line -match "3\.(10|11|12)\."
}

function Get-PythonExecutablePath {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd -and $pyCmd.Source -match "WindowsApps") {
        $pyCmd = $null
    }
    if ($pyCmd) {
        try {
            $ver = & $pyCmd.Source --version 2>&1 | Out-String
            if (Test-PythonVersionLine $ver) { return $pyCmd.Source }
        }
        catch { }
    }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($m in @(12, 11, 10)) {
            try {
                $ver = & py "-3.$m" --version 2>&1 | Out-String
                if ($LASTEXITCODE -eq 0 -and (Test-PythonVersionLine $ver)) {
                    $exe = (& py "-3.$m" -c "import sys; print(sys.executable)" 2>&1 | Select-Object -Last 1)
                    if ($exe -and (Test-Path -LiteralPath $exe)) { return $exe.Trim() }
                }
            }
            catch { }
        }
    }
    foreach ($folder in @("Python311", "Python312", "Python310", "Python313")) {
        $p = Join-Path $env:LocalAppData "Programs\Python\$folder\python.exe"
        if (Test-Path -LiteralPath $p) {
            try {
                $ver = & $p --version 2>&1 | Out-String
                if (Test-PythonVersionLine $ver) { return $p }
            }
            catch { }
        }
    }
    return $null
}

function Install-PythonWindows {
    param([int]$Minor)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $id = "Python.Python.3.$Minor"
        Write-Host "Python telepites winget-tel: $id ..." -ForegroundColor Cyan
        $p = Start-Process -FilePath $winget.Source -ArgumentList @(
            "install", "--id", $id, "-e", "--accept-package-agreements", "--accept-source-agreements"
        ) -Wait -PassThru -NoNewWindow
        if ($p.ExitCode -eq 0) { return $true }
        Write-Warning "winget lepett vissza (ExitCode: $($p.ExitCode)). Probalkozom python.org telepitvel..."
    }
    else {
        Write-Host "winget nincs — python.org telepito letoltese..." -ForegroundColor Yellow
    }
    $verMap = @{ 10 = "3.10.11"; 11 = "3.11.9"; 12 = "3.12.7" }
    if (-not $verMap.ContainsKey($Minor)) { $Minor = 11 }
    $fullVer = $verMap[$Minor]
    $url = "https://www.python.org/ftp/python/$fullVer/python-$fullVer-amd64.exe"
    $installer = Join-Path $env:TEMP ("python_install_{0}.exe" -f [Guid]::NewGuid().ToString("N"))
    Write-Host "Letoltes: $url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    try {
        Write-Host "Csendes telepites (per-user, PATH bovites)..." -ForegroundColor Cyan
        $args = @(
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0",
            "Include_doc=0", "Include_launcher=1", "SimpleInstall=1"
        )
        $p = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
        if ($p.ExitCode -ne 0) {
            throw "Python telepito hibakod: $($p.ExitCode)"
        }
    }
    finally {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    }
    return $true
}

function Install-GitWindows {
    if ($SkipGitInstall) {
        throw "Git nincs a PATH-on es -SkipGitInstall be van kapcsolva. Telepits Git-et vagy futtasd -UseZip kapcsoloval."
    }
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Git nincs a PATH-on es winget sem elerheto. Telepits Git-et (git-scm.com) vagy futtasd -UseZip kapcsoloval."
    }
    Write-Host "Git telepites winget-tel (Git.Git)..." -ForegroundColor Cyan
    $p = Start-Process -FilePath $winget.Source -ArgumentList @(
        "install", "--id", "Git.Git", "-e", "--accept-package-agreements", "--accept-source-agreements"
    ) -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) {
        throw "Git telepites sikertelen (ExitCode: $($p.ExitCode)). Telepits kezzel vagy hasznalj -UseZip."
    }
    Update-SessionPathFromMachineAndUser
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git telepites utan sem talalhato a PATH-on. Indits uj PowerShell ablakot."
    }
}

function Install-VCRedistIfNeeded {
    param([switch]$Skip)
    if ($Skip) { return }
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Host "winget nincs — VC++ Redist atugorva (OpenCV gond eseten telepits: VC++ 2015-2022 x64)." -ForegroundColor DarkGray
        return
    }
    Write-Host "VC++ 2015-2022 Redistributable (x64) — OpenCV/Torch DLL-ekhez..." -ForegroundColor Cyan
    $p = Start-Process -FilePath $winget.Source -ArgumentList @(
        "install", "--id", "Microsoft.VCRedist.2015+.x64", "-e", "--accept-package-agreements", "--accept-source-agreements"
    ) -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) {
        Write-Warning "VC++ Redist winget ExitCode: $($p.ExitCode) — folytatjuk; ha DLL hiba van, telepits kezzel."
    }
}

function New-FilteredRequirementsFile {
    param([string]$SourcePath, [string]$DestPath, [switch]$OmitTensorFlow)
    $lines = Get-Content -LiteralPath $SourcePath -Encoding UTF8
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($raw in $lines) {
        $trim = $raw.Trim()
        if ($trim.StartsWith("#") -or [string]::IsNullOrWhiteSpace($trim)) {
            $out.Add($raw)
            continue
        }
        $pkg = ($trim -split '\s+|;|#')[0].Trim().ToLowerInvariant()
        if ($pkg -eq "torch" -or $pkg -eq "torchvision") { continue }
        if ($OmitTensorFlow -and ($pkg -eq "tensorflow" -or $pkg.StartsWith("tensorflow-"))) { continue }
        $out.Add($raw)
    }
    $out | Set-Content -LiteralPath $DestPath -Encoding UTF8
}

# --- Repó gyökér meghatározása ---
$RepoRoot = $null

if ([string]::IsNullOrWhiteSpace($RepoUrl)) {
    $fromScript = Split-Path -Parent $PSScriptRoot
    if (Test-RepoRoot $fromScript) {
        $RepoRoot = (Resolve-Path $fromScript).Path
        Write-Host "Helyi repó: $RepoRoot" -ForegroundColor Cyan
    }
    elseif (Test-RepoRoot (Get-Location).Path) {
        $RepoRoot = (Get-Location).Path
        Write-Host "Helyi repó (cwd): $RepoRoot" -ForegroundColor Cyan
    }
    else {
        Write-Host @"
RepoUrl ures, es nem talalom a requirements_app.txt-t.

  1) Illeszd be a sajat repo linkjet (IDE ILLD BE, ha kesz a GitHub repo):
     .\scripts\install_dev.ps1 -RepoUrl 'https://github.com/TE/REPO'

     Git nelkul (ZIP):
     .\scripts\install_dev.ps1 -RepoUrl 'https://github.com/TE/REPO' -UseZip

  2) Vagy klónozd kezzel, lepj a repó gyökerébe, majd:
     .\scripts\install_dev.ps1
"@ -ForegroundColor Yellow
        exit 1
    }
}
else {
    if ([string]::IsNullOrWhiteSpace($TargetDir)) {
        $TargetDir = Join-Path (Get-Location).Path "DeepLabV3Plus-Pytorch"
    }
    $TargetDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($TargetDir)

    if (Test-Path -LiteralPath $TargetDir) {
        $children = Get-ChildItem -LiteralPath $TargetDir -Force -ErrorAction SilentlyContinue
        if ($children.Count -gt 0) {
            throw "Celkonyvtar nem ures: $TargetDir — torold vagy adj mas -TargetDir-t."
        }
    }
    else {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }

    if ($UseZip) {
        $zipUrl = Get-GitHubZipUrl -Url $RepoUrl -Br $Branch
        Write-Host "ZIP letoltes: $zipUrl" -ForegroundColor Cyan
        $zipFile = Join-Path $env:TEMP ("repo_{0}.zip" -f [Guid]::NewGuid().ToString("N"))
        try {
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile -UseBasicParsing
            Expand-Archive -LiteralPath $zipFile -DestinationPath $TargetDir -Force
        }
        finally {
            Remove-Item -LiteralPath $zipFile -Force -ErrorAction SilentlyContinue
        }
        # GitHub ZIP egy almappat hoz letre: Repo-branch
        $sub = Get-ChildItem -LiteralPath $TargetDir -Directory | Select-Object -First 1
        if (-not $sub) { throw "ZIP ures vagy hibas." }
        $inner = $sub.FullName
        Get-ChildItem -LiteralPath $inner | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination $TargetDir -Force
        }
        Remove-Item -LiteralPath $inner -Recurse -Force
    }
    else {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Install-GitWindows
        }
        Write-Host "git clone $RepoUrl (branch: $Branch) -> $TargetDir" -ForegroundColor Cyan
        git clone --depth 1 -b $Branch $RepoUrl $TargetDir
        if ($LASTEXITCODE -ne 0) { throw "git clone sikertelen. Probalj mas -Branch erteket (pl. master)." }
    }

    if (-not (Test-RepoRoot $TargetDir)) {
        throw "Letoltes utan nem talalom a requirements_app.txt-t: $TargetDir"
    }
    $RepoRoot = $TargetDir
}

Set-Location -LiteralPath $RepoRoot
Write-Host "Munkakonyvtar: $RepoRoot" -ForegroundColor Green

Install-VCRedistIfNeeded -Skip:$SkipVCRedist

# --- Python (hiany eseten telepites, majd PATH frissites) ---
Update-SessionPathFromMachineAndUser
$PythonExe = Get-PythonExecutablePath

if (-not $PythonExe) {
    if ($SkipPythonInstall) {
        throw "Python (3.10–3.12) nem talalhato es -SkipPythonInstall be van kapcsolva. Telepits kezzel vagy futtasd kapcsolo nelkul."
    }
    Write-Host "Python nem talalhato — telepites inditasa (3.$PythonMinor)..." -ForegroundColor Yellow
    $null = Install-PythonWindows -Minor $PythonMinor
    Update-SessionPathFromMachineAndUser
    $PythonExe = Get-PythonExecutablePath
}

if (-not $PythonExe) {
    throw "Python telepites utan sem talalhato. Indits uj PowerShell ablakot, vagy telepits kezzel 3.$PythonMinor-et es jelold be a PATH-hoz."
}

$ver = & $PythonExe --version 2>&1 | Out-String
Write-Host "Python: $($ver.Trim()) ($PythonExe)" -ForegroundColor Cyan
if ($ver -notmatch "3\.(10|11|12)\.") {
    Write-Warning "Javasolt: Python 3.11.x. Jelenlegi verzio eltérhet — lehetnek wheel/DLL problemak."
}

$venvPath = Join-Path $RepoRoot $VenvName
if (-not (Test-Path -LiteralPath $venvPath)) {
    Write-Host "venv letrehozasa: $venvPath" -ForegroundColor Cyan
    & $PythonExe -m venv $venvPath
}
else {
    Write-Host "Meglevo venv: $venvPath" -ForegroundColor Yellow
}

$activate = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path -LiteralPath $activate)) {
    throw "Nem talalom: $activate"
}
. $activate

python -m pip install --upgrade pip setuptools wheel
if ($GpuTorch) {
    $torchIndex = "https://download.pytorch.org/whl/cu124"
    Write-Host "PyTorch + torchvision (CUDA 12.4 wheel, NVIDIA driver szukseges)..." -ForegroundColor Cyan
}
else {
    $torchIndex = "https://download.pytorch.org/whl/cpu"
    Write-Host "PyTorch + torchvision (CPU wheel — tiszta Windows / driver nelkul)..." -ForegroundColor Cyan
}
python -m pip install torch torchvision --index-url $torchIndex

$reqMain = Join-Path $RepoRoot "requirements_app.txt"
$reqTmp = Join-Path $env:TEMP ("req_notorch_{0}.txt" -f [Guid]::NewGuid().ToString("N"))
try {
    New-FilteredRequirementsFile -SourcePath $reqMain -DestPath $reqTmp -OmitTensorFlow:$SkipTensorFlow
    Write-Host "pip install -r requirements_app.txt (torch mar telepitve$(if ($SkipTensorFlow) { ', tensorflow kihagyva' })) ..." -ForegroundColor Cyan
    python -m pip install -r $reqTmp
}
finally {
    Remove-Item -LiteralPath $reqTmp -Force -ErrorAction SilentlyContinue
}

Write-Host "pygrabber (DirectShow kamera lista, Windows)..." -ForegroundColor Cyan
python -m pip install pygrabber

if ($IncludeBuildDeps) {
    $buildReq = Join-Path $RepoRoot "deploy_exe\requirements_build.txt"
    if (Test-Path -LiteralPath $buildReq) {
        Write-Host "pip install -r deploy_exe\requirements_build.txt ..." -ForegroundColor Cyan
        python -m pip install -r $buildReq
    }
}

Write-Host ""
Write-Host "Import ellenorzes..." -ForegroundColor Cyan
python -c @"
import os, sys
sys.path.insert(0, os.getcwd())
import torch
import cv2
import customtkinter
import segmentation_models_pytorch
import network
print('OK - torch', torch.__version__, '| cuda', torch.cuda.is_available())
"@

Write-Host ""
Write-Host "KESZ." -ForegroundColor Green
Write-Host "  Aktivalas:  .\$VenvName\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  App futtatas:  python live_capture_app.py" -ForegroundColor White
Write-Host "  GPU PyTorch: futtasd ujra a scriptet -GpuTorch kapcsoloval (NVIDIA driver)." -ForegroundColor DarkGray
Write-Host "  Modellek: ha nincs a repoban, lasd deploy_exe\MODELLOK_MASOLASA.txt" -ForegroundColor DarkGray
