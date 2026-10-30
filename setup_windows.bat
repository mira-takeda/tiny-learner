@echo off
REM Setup + verificacion de tiny-learner en Windows (cmd).
REM Ejecutar desde la carpeta tiny-learner:  setup_windows.bat
python --version || (echo ERROR: python no encontrado. Instala Python 3.12+ de python.org. & exit /b 1)
if not exist repro_s16\repro.py (echo ERROR: corre este script desde la carpeta tiny-learner. & exit /b 1)
echo 1/3 Creando venv...
python -m venv .venv || exit /b 1
echo 2/3 Instalando dependencias...
.venv\Scripts\python -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python -m pip install -r repro_s16\requirements.txt || exit /b 1
echo 3/3 Verificando repro (~60-120 s)...
.venv\Scripts\python repro_s16\repro.py || exit /b 1
echo Listo. Esperado: A01 0.6467 / A-ONLY 0.1450 / B-ONLY 0.2458
