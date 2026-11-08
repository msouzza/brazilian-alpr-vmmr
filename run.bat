@echo off
REM ==============================================================================
REM Executavel Portatil ALPR & VMMR - Brazilian ALPR
REM Ativa automaticamente o ambiente virtual com aceleracao de GPU (CUDA)
REM ==============================================================================

cd /d "%~dp0"

IF EXIST "venv\Scripts\python.exe" (
    echo [INFO] Iniciando pipeline com ambiente acelerado (Python 3.12 + CUDA)...
    venv\Scripts\python.exe main.py
) ELSE (
    echo [INFO] Iniciando pipeline com Python padrao do sistema...
    python main.py
)

pause
