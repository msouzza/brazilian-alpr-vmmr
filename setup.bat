@echo off
REM ==============================================================================
REM Instalador / Configurador Portatil ALPR & VMMR
REM Cria o venv e instala as dependencias corretas para qualquer computador
REM ==============================================================================

cd /d "%~dp0"

echo [1/3] Verificando instalacao do Python 3.12...
py -3.12 --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [AVISO] Python 3.12 nao encontrado via py launcher. Tentando python padrao...
    python -m venv venv
) ELSE (
    echo [OK] Criando ambiente virtual com Python 3.12...
    py -3.12 -m venv venv
)

echo [2/3] Verificando placa de video NVIDIA (CUDA)...
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [OK] GPU NVIDIA detectada! Instalando PyTorch com suporte a CUDA 12.1...
    venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) ELSE (
    echo [INFO] Nenhuma GPU NVIDIA detectada. Instalando PyTorch otimizado para CPU...
    venv\Scripts\python.exe -m pip install torch torchvision
)

echo [3/3] Instalando dependencias do projeto...
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install "transformers<4.50" sentencepiece pip-system-certs

echo.
echo ==============================================================================
echo [SUCESSO] Configuracao concluida! Agora voce pode executar o run.bat
echo ==============================================================================
pause
