# Build dist\OpenVat-<version>-setup.exe on Windows.
# Needs: Python 3.10+ on PATH and Inno Setup 7 (https://jrsoftware.org/isdl.php,
# either edition; it installs alongside Inno Setup 6).  $env:ISCC may point at
# ISCC.exe if it is somewhere else.
# Run from a PowerShell prompt:   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# (On Linux, packaging/build_windows_wine.sh builds the same installer through Wine.)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Checked {
    # Run a program and stop if it fails (PowerShell 5 ignores native exit codes).
    $exe, $rest = $args
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "$exe failed (exit code $LASTEXITCODE)" }
}

$version = (Select-String -Path openvat\__init__.py -Pattern '^__version__ = "(.*)"').Matches[0].Groups[1].Value

Write-Host "== PyInstaller"
Invoke-Checked python -m pip install -q -r requirements.txt pyinstaller
Invoke-Checked python -m PyInstaller --noconfirm --clean --distpath build\windows\dist --workpath build\windows\work packaging\openvat.spec

Write-Host "== Inno Setup 7"
$candidates = @(
    $env:ISCC,
    "$env:ProgramFiles\Inno Setup 7\ISCC.exe",                       # 64-bit edition, all users
    "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",                # 32-bit edition, all users
    "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",              # current user only
    "$env:LOCALAPPDATA\Programs\Inno Setup 7 (32-bit)\ISCC.exe"
)
$iscc = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup 7 (ISCC.exe) not found. Install it from https://jrsoftware.org/isdl.php, or set `$env:ISCC."
}
Invoke-Checked $iscc "/DAppVersion=$version" packaging\installer.iss
Write-Host "built dist\OpenVat-$version-setup.exe"
