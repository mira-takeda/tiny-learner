# Setup + verificación de tiny-learner en Windows.
# Ejecutar desde la carpeta tiny-learner:  .\setup_windows.ps1
$ErrorActionPreference = "Stop"

$v = & python --version 2>&1
Write-Host "Interprete: $v"
$nums = [regex]::Match($v, '(\d+)\.(\d+)')
if (-not $nums.Success -or [int]$nums.Groups[1].Value -lt 3 -or ([int]$nums.Groups[1].Value -eq 3 -and [int]$nums.Groups[2].Value -lt 12)) {
  Write-Host "ERROR: se necesita Python 3.12+ (ideal 3.13/3.14). Instala uno reciente de python.org y reintenta." -ForegroundColor Red
  exit 1
}
if (-not (Test-Path "repro_s16\repro.py")) {
  Write-Host "ERROR: corre este script desde la carpeta tiny-learner (no encuentro repro_s16\repro.py)." -ForegroundColor Red
  exit 1
}

Write-Host "1/3 Creando venv..."
python -m venv .venv
Write-Host "2/3 Instalando dependencias..."
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r repro_s16\requirements.txt
Write-Host "3/3 Verificando repro (~60-120 s)..."
.venv\Scripts\python repro_s16\repro.py
Write-Host "Listo. Esperado: A01 0.6467 / A-ONLY 0.1450 / B-ONLY 0.2458" -ForegroundColor Green
