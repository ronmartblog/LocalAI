@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: -- Off-profile cache redirection (v5.3.10) --
:: Same redirection block as setup.bat / run.bat / fix_nvidia_pytorch.bat: the
:: torch + torch-directml wheels are large and MUST land on the install drive,
:: not a tiny profile drive. Set BEFORE the python_path.bat call so pip never
:: sees the original.
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
echo   Fix PyTorch for DirectML (AMD / Intel / AI PC)
echo   Use this when image generation fails with errors like
echo   "Entry Point Not Found" or "torch_library_impl could not
echo   be located in _torchaudio.pyd" - it means torch and
echo   torchaudio drifted out of ABI sync.
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

:: Refuse to run on Windows ARM64 - torch-directml has no ARM64 wheel
"%LOCALAI_PYTHON%" -c "import platform, sys; sys.exit(0 if platform.machine().upper() == 'ARM64' else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Windows ARM64 / Snapdragon X detected.
    echo         torch-directml does not have an ARM64 wheel on PyPI,
    echo         so DirectML image generation cannot run on this system.
    echo         Chat models, vision, ONNX and embeddings still work on CPU.
    pause
    exit /b 1
)

:: Warn if NVIDIA GPU is present - this is the wrong script for that case
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] NVIDIA GPU detected!
    echo           For NVIDIA hardware, use fix_nvidia_pytorch.bat instead.
    echo           DirectML works on NVIDIA but is slower than CUDA.
    echo.
    set /p CONTINUE="Continue with DirectML install anyway? [y/N]: "
    if /i not "!CONTINUE!"=="y" (
        echo Cancelled.
        pause
        exit /b 0
    )
)

echo [*] This will:
echo     1. Uninstall current PyTorch ^(CUDA, DirectML, or CPU-only^)
echo     2. Reinstall torch-directml + matched torch / torchvision / torchaudio
echo     3. Verify the DirectML device is visible to PyTorch
echo.
echo NOTE: This recovers from torch/torchaudio version drift on Intel / AMD /
echo       AI PC systems. Image generation will use the integrated GPU
echo       via DirectML and is significantly slower than dedicated GPUs.
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
echo [*] Installing torch-directml ^(pins torch^) ...
echo     This downloads large wheels and can take several minutes.
echo     Progress will be shown below; do not close this window.
:: v5.3.10: PIP_CACHE_DIR / TMP / TEMP have been redirected to %~dp0.cache\
:: by the env-block at the top of this script so the multi-GB wheel can no
:: longer spool into a small profile drive and fail with OSError 28.
:: v5.5.11 (Ron, 2026-05-26): two-step install to prevent torchaudio drift.
:: v5.5.10 and earlier used `pip install --upgrade torch-directml torch
:: torchvision torchaudio` which let pip's resolver pick the LATEST torchaudio
:: (e.g. 2.11.0) against torch-directml's pinned torch (2.4.1), producing
:: WinError 127 / "torch_library_impl in _torchaudio.pyd" on Intel AI PC.
:: Step 1 installs torch-directml ALONE so its torch pin lands first. Step 2
:: discovers that pinned torch version at runtime and installs
:: torchaudio==<same-version> + torchvision against it.
"%LOCALAI_PYTHON%" -m pip install --upgrade --no-input --disable-pip-version-check torch-directml

if errorlevel 1 (
    echo.
    echo [ERROR] torch-directml install failed.
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
echo [*] Discovering pinned torch version ...
set "VFILE=%TEMP%\__localai_torchver_fix.txt"
set "TORCH_VER="
"%LOCALAI_PYTHON%" -c "import torch; print(torch.__version__.split(chr(43))[0])" > "%VFILE%" 2>nul
if exist "%VFILE%" set /p TORCH_VER=<"%VFILE%"
if exist "%VFILE%" del "%VFILE%" >nul 2>&1
if not defined TORCH_VER (
    echo.
    echo [ERROR] Could not discover installed torch version after torch-directml install.
    pause
    exit /b 1
)
echo [*] torch pinned to %TORCH_VER%; installing matched torchaudio + torchvision ...
"%LOCALAI_PYTHON%" -m pip install --no-input --disable-pip-version-check "torchaudio==%TORCH_VER%" torchvision

if errorlevel 1 (
    echo.
    echo [ERROR] Matched torchaudio==%TORCH_VER% / torchvision install failed.
    echo         Attempting to recover any partial pip cache ...
    "%LOCALAI_PYTHON%" -m pip cache purge >nul 2>&1
    pause
    exit /b 1
)

echo.
echo [*] Testing DirectML and torchaudio ABI alignment...
REM v5.5.11 (Ron, 2026-05-26 SQT nit): wrap the verify import with
REM SetErrorMode(0x8001) (SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX)
REM to suppress the Windows "WinError 127" popup dialog if a stale
REM _torchaudio.pyd somehow still mismatches after the matched-set
REM reinstall. Mirrors the same pattern in setup.bat:install_directml_pytorch.
"%LOCALAI_PYTHON%" -c "import ctypes; ctypes.windll.kernel32.SetErrorMode(0x8001); import torch, torchaudio, torch_directml; print(f'PyTorch: {torch.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'torch-directml available: {torch_directml.is_available()}'); print(f'DirectML device: {torch_directml.device_name(0) if torch_directml.is_available() else \"unavailable\"}'); assert torch_directml.is_available(), 'torch-directml installed but no DirectML device is visible'" 2>&1

if errorlevel 1 (
    echo.
    echo [WARNING] DirectML test failed. Check the GPU driver, then run this script again.
) else (
    echo.
    echo ============================================================
    echo   PyTorch with DirectML installed successfully!
    echo.
    echo   Your integrated GPU is now ready for use.
    echo   Restart LocalAI to use DirectML acceleration.
    echo.
    echo   Note: Image generation on integrated GPUs is significantly
    echo         slower than dedicated GPUs. Stick to smaller models
    echo         like Realistic Vision V6.
    echo ============================================================
)

echo.
pause
