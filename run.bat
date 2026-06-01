@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
cd /d "%~dp0"
title *** DO NOT CLOSE THIS WINDOW - LocalAI Studio is running ***

:: ── Off-profile cache redirection (v5.3.10) ───────────────────────────────────
:: Mirror setup.bat: route pip / HuggingFace / torch / TMP caches to
:: <app>\.cache\ and OLLAMA_MODELS to <app>\Ollama\ BEFORE python starts so
:: runtime imports and any in-app installer can't spill into a roamed /
:: roaming-profile-capped drive.
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

:: ── Locate Python ─────────────────────────────────────────────────────────────
set PYTHON_EXE=
if exist "%~dp0python_path.bat" (
    call "%~dp0python_path.bat"
    if defined LOCALAI_PYTHON if exist "%LOCALAI_PYTHON%" set PYTHON_EXE=%LOCALAI_PYTHON%
)
if not defined PYTHON_EXE (
    for %%P in (
        "%LocalAppData%\Programs\Python\Python313\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python311\python.exe"
    ) do (
        if exist %%P (
            set PYTHON_EXE=%%~P
            goto :python_found
        )
    )
    :: Fall back to PATH
    where python >nul 2>&1 && set PYTHON_EXE=python
)
:python_found
if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Run setup.bat first.
    pause
    exit /b 1
)

:: ── Locate and start Ollama ───────────────────────────────────────────────────
set OLLAMA_EXE=
for %%O in (
    "%LocalAppData%\Programs\Ollama\ollama.exe"
    "%ProgramFiles%\Ollama\ollama.exe"
) do (
    if exist %%O set OLLAMA_EXE=%%~O
)
if defined OLLAMA_EXE (
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
    if errorlevel 1 (
        echo [*] Starting Ollama in background ...
        start "" /b "%OLLAMA_EXE%" serve >nul 2>&1
    )
) else (
    where ollama >nul 2>&1
    if not errorlevel 1 (
        tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
        if errorlevel 1 start "" /b ollama serve >nul 2>&1
    ) else (
        echo [WARNING] Ollama not found. GPU/CPU models require Ollama.
        echo           Download from: https://ollama.com/download/windows
    )
)

:: ── ComfyUI auto-start (handled by app now) ──────────────────────────────────
:: The app will auto-detect GPU and start ComfyUI with the right flags
:: when Generate, Analyze to Prompt, or Restart needs the backend

:: ── DirectML ABI preflight (v5.5.10) ──────────────────────────────────────────
:: Catch the "Entry Point Not Found: torch_library_impl in _torchaudio.pyd"
:: Windows dialog BEFORE the app launches. SetErrorMode(0x8001) suppresses
:: the OS popup so the failure surfaces as a clean exit code instead of a
:: blocking dialog the user can't dismiss without losing app state. Only
:: runs when torch-directml is installed (skipping CUDA-only / CPU-only
:: setups and Snapdragon ARM64 which never installs torch-directml).
"%PYTHON_EXE%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch_directml') else 2)" >nul 2>&1
if errorlevel 1 goto :skip_dml_preflight
"%PYTHON_EXE%" -c "import ctypes, sys; ctypes.windll.kernel32.SetErrorMode(0x8001); import torch, torchaudio, torch_directml" >nul 2>&1
if not errorlevel 1 goto :skip_dml_preflight
echo.
echo [WARNING] DirectML PyTorch ABI drift detected.
echo           torch and torchaudio are out of sync. Image generation will
echo           crash with "Entry Point Not Found: torch_library_impl in
echo           _torchaudio.pyd".
echo.
echo           To repair: run fix_directml_pytorch.bat, then re-launch.
echo.
:: Pre-clear CONTINUE so an inherited environment value (e.g. CONTINUE=y from
:: a parent shell) cannot silently bypass the warning prompt.
set "CONTINUE="
set /p CONTINUE="Launch LocalAI anyway (image generation will fail)? [y/N]: "
if /i not "%CONTINUE%"=="y" exit /b 1
:skip_dml_preflight

:: ── Launch app ────────────────────────────────────────────────────────────────
:: v2026.06.01.4: write session-boundary markers to localai.log so the file
:: ALWAYS exists when the error message below references it. The app's own
:: logger appends to the same file once App.__init__ runs (line ~937 of
:: src/app.py). No tee here — that would race with the app logger's writes.
echo. >> "%~dp0localai.log" 2>nul
echo === LocalAI run.bat session start %DATE% %TIME% === >> "%~dp0localai.log" 2>nul
"%PYTHON_EXE%" main.py
set RUN_EXIT_CODE=%errorlevel%
echo === LocalAI run.bat session end %DATE% %TIME% (exit %RUN_EXIT_CODE%) === >> "%~dp0localai.log" 2>nul
if not "%RUN_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] App exited with code %RUN_EXIT_CODE%.
    echo         App-level events are logged to:
    echo             %~dp0localai.log
    echo         If the crash happened at launch ^(module import error,
    echo         broken Python package^), the Traceback printed above is
    echo         the best diagnostic ^- the app logger never got a chance
    echo         to start. Also check %~dp0setup.log
    echo         to see what your install put on the machine.
    pause
)
