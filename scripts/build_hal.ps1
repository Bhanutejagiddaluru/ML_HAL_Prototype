
Param()

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Join-Path $Here ".."
$ExtDir = Join-Path $Root "src\hal\cpp"
$BuildDir = Join-Path $ExtDir "build"

if (!(Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }
Set-Location $BuildDir
cmake -DPYBIND11_FINDPYTHON=ON ..
cmake --build . --config Release

# Install to user site-packages so Python can 'import hal_ext'
$site = python - << 'PY'
import site; print(site.getusersitepackages())
PY

$site = $site.Trim()
New-Item -ItemType Directory -Force -Path $site | Out-Null

Get-ChildItem -Recurse -Filter "hal_ext*.pyd" | ForEach-Object {
    Copy-Item $_.FullName -Destination $site -Force
}
Write-Host "Installed hal_ext to: $site"
