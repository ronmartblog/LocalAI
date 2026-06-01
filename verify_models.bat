@echo off
:: LocalAI Studio - Model Integrity Verifier
:: Checks every downloaded model against its expected file size.
:: Truncated or corrupted files are deleted so they can be re-downloaded.
setlocal

:: ── Off-profile cache redirection (v5.3.10) ───────────────────────────────────
:: Same env block as setup.bat / run.bat so verify_models.py and any pip-driven
:: re-download path it triggers uses the on-install-drive cache, not the
:: profile drive.
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

echo.
echo ============================================================
echo   LocalAI Studio -- Model Integrity Check
echo ============================================================
echo.

:: ── Find Python ───────────────────────────────────────────────────────────────
set "PYTHON_EXE="
if exist "%~dp0python_path.bat" call "%~dp0python_path.bat"
if defined LOCALAI_PYTHON set "PYTHON_EXE=%LOCALAI_PYTHON%"
if not defined PYTHON_EXE for /f "tokens=*" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Run setup.bat first.
    pause
    exit /b 1
)

:: ── Run the verifier ──────────────────────────────────────────────────────────
"%PYTHON_EXE%" "%~dp0verify_models.py"
set EXIT=%ERRORLEVEL%

if %EXIT% neq 0 (
    echo.
    echo [ERROR] Verification failed unexpectedly.
)
pause
exit /b %EXIT%
