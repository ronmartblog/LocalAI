@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ── Off-profile cache redirection (v5.3.10) ───────────────────────────────────
:: Same redirection block as setup.bat / run.bat: the 2.8 GB CUDA wheel that
:: this script downloads MUST land on the install drive, not a tiny profile
:: drive. Set BEFORE the python_path.bat call so pip never sees the original.
set "PIP_CACHE_DIR=%~dp0.cache\pip"
set "TMP=%~dp0.cache\tmp"
set "TEMP=%~dp0.cache\tmp"
set "HF_HOME=%~dp0.cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0.cache\huggingface\hub"
set "TORCH_HOME=%~dp0.cache\torch"
set "XDG_CACHE_HOME=%~dp0.cache"
set "OLLAMA_MODELS=%~dp0Ollama"
if not exist "%~dp0.cache\pip" mkdir "%~dp0.cache\pip" >nul 2>&1
if not exist "%~dp0.cache\tmp" mkdir "%~dp0.cache\tmp" >nul 2>&1
if not exist "%~dp0.cache\huggingface\hub" mkdir "%~dp0.cache\huggingface\hub" >nul 2>&1
if not exist "%~dp0.cache\torch" mkdir "%~dp0.cache\torch" >nul 2>&1
if not exist "%~dp0Ollama" mkdir "%~dp0Ollama" >nul 2>&1

echo ============================================================
echo   Fix PyTorch for NVIDIA GPU (CUDA)
echo   Use this when PyTorch is CPU-only or DirectML on
echo   a machine with NVIDIA GPU
echo ============================================================
echo.

:: Load Python
call "%~dp0python_path.bat"
if not defined LOCALAI_PYTHON (
    echo [ERROR] Python path not configured. Run setup.bat first.
    pause
    exit /b 1
)

echo [*] Python: %LOCALAI_PYTHON%
echo.

:: Check if NVIDIA GPU is present
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] No NVIDIA GPU detected!
    echo           This script is for NVIDIA GPU machines only.
    echo.
    set /p CONTINUE="Continue anyway? [y/N]: "
    if /i not "!CONTINUE!"=="y" (
        echo Cancelled.
        pause
        exit /b 0
    )
)

echo [*] This will:
echo     1. Uninstall current PyTorch (DirectML or CPU-only)
echo     2. Install PyTorch with CUDA 12.8 support
echo     3. Test CUDA availability
echo.
echo NOTE: If you see "CUDA error: operation not supported" during image generation,
echo       this upgrade fixes it. Requires CUDA 12.8+ driver on the machine.
echo.
set /p CONFIRM="Continue? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo [*] Uninstalling current PyTorch...
"%LOCALAI_PYTHON%" -m pip uninstall -y torch torchvision torchaudio torch-directml

echo.
echo [*] Installing PyTorch with CUDA 12.8 support...
echo     This downloads large CUDA wheels and can take several minutes.
echo     Progress will be shown below; do not close this window.
:: v5.3.10: PIP_CACHE_DIR / TMP / TEMP have been redirected to %~dp0.cache\
:: by the env-block at the top of this script so the multi-GB wheel can no
:: longer spool into a small profile drive and fail with OSError 28. The old
:: --no-cache-dir guard is therefore obsolete.
"%LOCALAI_PYTHON%" -m pip install --upgrade --no-input --disable-pip-version-check torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed.
    echo         The pip cache + TMP have been redirected to %~dp0.cache\.
    echo         If you still saw OSError 28, free space on the install drive
    echo         or check that nothing ^(antivirus, group policy^) is locking
    echo         files under %~dp0.cache\.
    echo         Attempting to recover any partial pip cache ...
    "%LOCALAI_PYTHON%" -m pip cache purge >nul 2>&1
    pause
    exit /b 1
)

echo.
echo [*] Testing CUDA...
"%LOCALAI_PYTHON%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}'); assert torch.cuda.is_available(), 'CUDA is not available after install'; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>&1

if errorlevel 1 (
    echo.
    echo [WARNING] CUDA test failed. Check the NVIDIA driver version, then run this script again.
) else (
    echo.
    echo ============================================================
    echo   PyTorch with CUDA 12.8 installed successfully!
    echo.
    echo   Your NVIDIA GPU is now ready for use.
    echo   Restart LocalAI to use CUDA acceleration.
    echo ============================================================
)

echo.
pause
