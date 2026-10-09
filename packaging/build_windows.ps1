# Build dist\OpenVat-<version>-setup.exe on Windows.
# Needs: Python 3.10+ on PATH and Inno Setup 6 (https://jrsoftware.org/isinfo.php).
# Run from a PowerShell prompt:   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$version = python -c "import openvat; print(openvat.__version__)"

Write-Host "== PyInstaller"
python -m pip install -q -r requirements.txt pyinstaller
python packaging\make_icon.py packaging\openvat.ico
python -m PyInstaller --noconfirm --clean packaging\openvat.spec

Write-Host "== Inno Setup"
$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "ISCC.exe") |
    Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isinfo.php" }
& $iscc "/DAppVersion=$version" packaging\installer.iss
Write-Host "built dist\OpenVat-$version-setup.exe"
