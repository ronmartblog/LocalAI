@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
:: NOTE: The v2026.06.01.4 self-tee wrapper was REMOVED in v2026.06.01.5 — it broke
:: interactive `set /p` prompts because piping cmd's stdout through PowerShell
:: switches Windows to block buffering, so prompt text never flushes before user
:: input is required. setup.log diagnostic capture is deferred to a future design
:: that does not break interactivity (e.g. write-on-failure only, or a native
:: Windows transcript mechanism that preserves console I/O). See AGENTS.md
:: "DO NOT REGRESS" row for the full rationale.
setlocal enabledelayedexpansion

:: ── Off-profile cache redirection (v5.3.10) ───────────────────────────────────
:: Route pip download/build, HuggingFace, torch, and TMP caches to <app>\.cache
:: BEFORE any pip / python invocation so multi-GB CUDA wheels never spool into
:: %USERPROFILE%\AppData or %TEMP% on size-capped profile drives.
:: %~dp0 always ends with a backslash; we keep the literal as-is here because
:: it concatenates cleanly with the appended subfolder name.
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
echo   LocalAI Studio -- First-Time Setup
echo ============================================================
echo.

set SETUP_ALL_FEATURES=%SETUP_ALL_FEATURES%
if /i "%SETUP_SKIP_OPTIONAL_PROMPTS%"=="1" set SETUP_ALL_FEATURES=n
if /i "%SETUP_NONINTERACTIVE%"=="1" if not defined SETUP_ALL_FEATURES set SETUP_ALL_FEATURES=y
if not defined SETUP_ALL_FEATURES set /p SETUP_ALL_FEATURES="Setup all features? [Y/n]: "
if "!SETUP_ALL_FEATURES!"=="" set SETUP_ALL_FEATURES=y
if /i "!SETUP_ALL_FEATURES!"=="yes" set SETUP_ALL_FEATURES=y
if /i "!SETUP_ALL_FEATURES!"=="y" (
    set SETUP_ALL_FEATURES=1
) else (
    set SETUP_ALL_FEATURES=0
)
call :detect_setup_hardware
if "!SETUP_ALL_FEATURES!"=="1" (
    echo [*] All applicable features will be installed.
    if "!SETUP_HAS_GPU!"=="1" (
        echo [*] GPU acceleration features apply to: !SETUP_GPU_NAMES!
        if "!SETUP_HAS_NVIDIA!"=="1" echo [*] NVIDIA CUDA acceleration applies to: !SETUP_NVIDIA_GPU_NAMES!
        if "!SETUP_HAS_DML_GPU!"=="1" echo [*] DirectML acceleration applies to: !SETUP_DML_GPU_NAMES!
    ) else (
        echo [*] No supported local GPU detected; GPU-only acceleration will be skipped.
    )
    echo.
)

:: ── Pre-flight: Check Windows Installer availability ─────────────────────────
:: Warn early if another installation is running to save time
for /f %%i in ('tasklist /FI "IMAGENAME eq msiexec.exe" 2^>nul ^| find /c /i "msiexec.exe"') do set PREFLIGHT_MSI=%%i
if %PREFLIGHT_MSI% gtr 0 (
    echo [WARNING] Detected %PREFLIGHT_MSI% active Windows Installer process^(es^).
    echo           Setup may fail if another installation is in progress.
    echo.
    if /i "%SETUP_NONINTERACTIVE%"=="1" (
        set CONTINUE_ANYWAY=y
    ) else (
        set /p CONTINUE_ANYWAY="Continue anyway? [y/N]: "
    )
    if /i not "!CONTINUE_ANYWAY!"=="y" (
        echo [*] Please complete or cancel any running installations, then re-run setup.bat
        pause
        exit /b 0
    )
    echo.
)

:: ── 1. Python check / install ────────────────────────────────────────────────

:: Helper: find any Python 3.x installation in the standard locations
call :find_python
if defined PYTHON_EXE goto :found_python

:: Not found on PATH or known paths — install via winget
echo [*] Python not found. Installing via winget ...

:: Retry up to 3 times if Windows Installer is busy (error 1618)
set INSTALL_ATTEMPTS=0
:retry_python_install
set /a INSTALL_ATTEMPTS+=1

winget install --id Python.Python.3.13 --source winget --silent --accept-package-agreements --accept-source-agreements
set WINGET_EXIT=%ERRORLEVEL%

:: Error 1618 = another installation in progress
if %WINGET_EXIT% equ 1618 (
    if %INSTALL_ATTEMPTS% lss 3 (
        echo [*] Windows Installer is busy. Waiting 15 seconds before retry %INSTALL_ATTEMPTS%/3 ...
        timeout /t 15 /nobreak >nul
        goto :retry_python_install
    ) else (
        echo.
        echo [ERROR] Windows Installer is still busy after 3 attempts.
        echo.
        call :diagnose_msi_service
        pause
        exit /b 1
    )
)

:: Try Python 3.12 if 3.13 fails for other reasons
if %WINGET_EXIT% neq 0 (
    echo [*] Python 3.13 unavailable, trying 3.12 ...
    winget install --id Python.Python.3.12 --source winget --silent --accept-package-agreements --accept-source-agreements
    set WINGET_EXIT=%ERRORLEVEL%

    if !WINGET_EXIT! equ 1618 (
        echo [ERROR] Windows Installer is busy. Please wait for any pending installations
        echo         to complete, then re-run this script.
        pause
        exit /b 1
    )
)

:: Give the installer time to complete and write files
echo [*] Waiting for installer to complete ...
timeout /t 10 /nobreak >nul

:: Refresh PATH from registry into this session so newly installed tools are visible
call :refresh_path

:: Try finding Python again after install
call :find_python
if not defined PYTHON_EXE (
    echo.
    echo [ERROR] Python installation failed or was installed to an unexpected location.
    echo         Please install Python manually from https://www.python.org/downloads/
    echo         then re-run this script.
    pause
    exit /b 1
)

:found_python
:: Verify the exe actually works
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Found Python at %PYTHON_EXE% but it does not run correctly.
    echo         Try deleting it and re-running setup.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYVER=%%v
echo [OK] %PYVER%  ^(%PYTHON_EXE%^)
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required. Detected: %PYVER%
    echo         Install Python 3.10 or newer, then re-run setup.bat.
    pause
    exit /b 1
)

:: Save python path for run.bat — no trailing space before >
(echo set LOCALAI_PYTHON=%PYTHON_EXE%)> "%~dp0python_path.bat"

:: ── 2. pip upgrade ───────────────────────────────────────────────────────────
echo.
echo [*] Upgrading pip ...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] pip upgrade failed — continuing with existing pip version.
)

:: Older uninstall.bat versions could remove gguf's transitive deps while
:: leaving gguf itself installed. Repair that before the first normal pip
:: install so setup does not start with resolver-conflict noise.
"%PYTHON_EXE%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('gguf') is not None else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [*] Repairing existing GGUF support dependencies ...
    "%PYTHON_EXE%" -m pip install --upgrade --no-input --disable-pip-version-check --no-warn-conflicts "PyYAML>=5.1" "tqdm>=4.27" --quiet
    if errorlevel 1 (
        echo [WARNING] Could not repair GGUF support dependencies yet; image setup will retry later.
    )
)

:: ── 3. Core dependencies ─────────────────────────────────────────────────────
echo [*] Installing required packages (customtkinter, requests, psutil) ...
"%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts customtkinter requests psutil --quiet
if errorlevel 1 (
    echo [ERROR] Package installation failed. Check internet connection and try again.
    pause
    exit /b 1
)

:: Verify imports actually work — pip exit 0 does not guarantee a usable install
"%PYTHON_EXE%" -c "import customtkinter, requests, psutil" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Packages installed but imports failed. Re-running pip with full output:
    "%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts customtkinter requests psutil
    pause
    exit /b 1
)
echo [OK] Core packages installed and verified.

:: ── 4. Optional ONNX/NPU support ─────────────────────────────────────────────
echo.
if "!SETUP_ALL_FEATURES!"=="1" (
    set INSTALL_ONNX=y
    echo [*] Setup all features: installing ONNX/DirectML runtime support.
) else (
    set /p INSTALL_ONNX="Install NPU/DirectML (ONNX Runtime) support? [y/N]: "
)
if /i not "!INSTALL_ONNX!"=="y" goto :skip_onnx

echo [*] Installing ONNX runtime packages (variant: !SETUP_ONNX_LABEL!) ...
:: v2026.06.01.7 (Ron, 2026-06-01): branch on GPU vendor and install exactly
:: ONE onnxruntime runtime + ONE onnxruntime-genai variant. The three runtime
:: packages (onnxruntime / onnxruntime-gpu / onnxruntime-directml) and the
:: three genai packages (-genai / -genai-cuda / -genai-directml) all install
:: into the same onnxruntime/ namespace -- installing more than one leaves
:: onnxruntime.__file__ == None and InferenceSession gone (workstation-class
:: Py3.13 regression that broke Toolbox Speak in v.6). Pre-purge ALL six variants
:: before each install so half-installed state from any earlier run cannot
:: shadow the chosen wheel.
"%PYTHON_EXE%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-genai onnxruntime-genai-cuda onnxruntime-genai-directml >nul 2>&1
"%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts !SETUP_ONNX_PKG! !SETUP_ONNX_GENAI_PKG! "optimum[onnxruntime]" transformers "huggingface-hub>=0.34.0,<1.0" --quiet
if errorlevel 1 (
    echo [WARNING] Some ONNX packages failed. App still works with Ollama backend.
) else (
    echo [OK] ONNX/NPU packages installed.
    "%PYTHON_EXE%" -c "from onnxruntime import InferenceSession; import onnxruntime as ort; assert ort.__file__ is not None, 'onnxruntime namespace is broken (mutually exclusive variants collided)'; providers=ort.get_available_providers(); assert '!SETUP_ONNX_EP!' in providers, providers" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] !SETUP_ONNX_LABEL! ONNX provider was not detected after install.
    ) else (
        echo [OK] !SETUP_ONNX_LABEL! ONNX provider verified.
    )
)
goto :done_onnx

:skip_onnx
echo [*] Skipped ONNX. You can install it later from the Settings page.
:done_onnx

echo.
if "!SETUP_ALL_FEATURES!"=="1" (
    set INSTALL_OPENVINO=y
    echo [*] Setup all features: installing OpenVINO GenAI support.
) else (
    set /p INSTALL_OPENVINO="Install OpenVINO GenAI support? [y/N]: "
)
if /i not "!INSTALL_OPENVINO!"=="y" goto :done_openvino
echo [*] Installing OpenVINO GenAI packages ...
"%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts openvino openvino-genai --quiet
if errorlevel 1 (
    echo [WARNING] Some OpenVINO packages failed. App still works with Ollama backend.
) else (
    echo [OK] OpenVINO GenAI packages installed.
)
:done_openvino

echo.
if "!SETUP_ALL_FEATURES!"=="1" (
    set INSTALL_UTILITY=y
    echo [*] Setup all features: installing OCR/speech/embedding demo support.
) else (
    set /p INSTALL_UTILITY="Install OCR/speech/embedding demo support? [y/N]: "
)
if /i not "!INSTALL_UTILITY!"=="y" goto :done_utility
echo [*] Installing utility model demo packages ...
"%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts torch torchvision transformers accelerate safetensors sentence-transformers soundfile scipy pillow timm sentencepiece librosa backoff onnxruntime "optimum[onnxruntime]" diffusers pyttsx3 piper-tts einops peft "huggingface-hub>=0.34.0,<1.0" hf_xet --quiet
if errorlevel 1 (
    echo [WARNING] Some utility demo packages failed. Chat and image generation still work.
) else (
    echo [OK] Utility demo packages installed.
)
if /i "!INSTALL_ONNX!"=="y" (
    echo [*] Reconciling !SETUP_ONNX_LABEL! ONNX runtime after utility packages ...
    :: v2026.06.01.7 (Ron, 2026-06-01): utility install above pulls bare
    :: `onnxruntime` (CPU) via dep extras like optimum[onnxruntime]; on
    :: NVIDIA / AMD-Intel boxes this would shadow the GPU / DML wheel we
    :: chose in the ONNX install step, leaving onnxruntime.__file__ == None
    :: and InferenceSession gone. Purge ALL six variants and re-install the
    :: chosen one so it definitively wins. cd back to the script dir first
    :: in case any earlier step left cwd on a now-missing drive (a
    :: workstation-class setup log in v.6 showed two spurious "The system
    :: cannot find the drive specified." messages right here -- pinning
    :: cwd suppresses them).
    cd /d "%~dp0"
    "%PYTHON_EXE%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-genai onnxruntime-genai-cuda onnxruntime-genai-directml >nul 2>&1
    "%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts !SETUP_ONNX_PKG! !SETUP_ONNX_GENAI_PKG! --quiet
    if errorlevel 1 (
        echo [WARNING] !SETUP_ONNX_LABEL! ONNX runtime re-install failed. ONNX/NPU acceleration may be unavailable.
    ) else (
        "%PYTHON_EXE%" -c "from onnxruntime import InferenceSession; import onnxruntime as ort; assert ort.__file__ is not None, 'onnxruntime namespace is broken (mutually exclusive variants collided)'; providers=ort.get_available_providers(); assert '!SETUP_ONNX_EP!' in providers, providers" >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] !SETUP_ONNX_LABEL! ONNX provider missing or onnxruntime namespace is broken after utility install.
            echo         Toolbox features that import InferenceSession will fail.
            echo         Re-run setup.bat to retry, or manually:
            echo           "%PYTHON_EXE%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-genai onnxruntime-genai-cuda onnxruntime-genai-directml
            echo           "%PYTHON_EXE%" -m pip install !SETUP_ONNX_PKG! !SETUP_ONNX_GENAI_PKG!
        ) else (
            echo [OK] !SETUP_ONNX_LABEL! ONNX provider verified after utility install.
        )
    )
)
:done_utility

:: ── 5. Ollama check / install ─────────────────────────────────────────────────
echo.

:: Check known install paths first — don't rely on stale session PATH
set OLLAMA_EXE=
if exist "%LocalAppData%\Programs\Ollama\ollama.exe" set OLLAMA_EXE=%LocalAppData%\Programs\Ollama\ollama.exe
if exist "%ProgramFiles%\Ollama\ollama.exe"          set OLLAMA_EXE=%ProgramFiles%\Ollama\ollama.exe
if not defined OLLAMA_EXE (
    :: Fall back to PATH search (works on run 2+)
    for /f "tokens=*" %%O in ('where ollama 2^>nul') do if not defined OLLAMA_EXE set OLLAMA_EXE=%%O
)

if defined OLLAMA_EXE (
    for /f "tokens=*" %%v in ('"!OLLAMA_EXE!" --version 2^>^&1') do set OLLAMAVER=%%v
    echo [OK] Ollama: !OLLAMAVER!  ^(!OLLAMA_EXE!^)
) else (
    echo [*] Ollama not found. Installing via winget ...
    winget install --id Ollama.Ollama --source winget --silent --accept-package-agreements --accept-source-agreements
    set OLLAMA_EXIT=%ERRORLEVEL%

    if !OLLAMA_EXIT! equ 1618 (
        echo [WARNING] Windows Installer is busy. Skipping Ollama auto-install.
        echo           Download manually: https://ollama.com/download/windows
    ) else if !OLLAMA_EXIT! neq 0 (
        echo [WARNING] Could not auto-install Ollama.
        echo           Download manually: https://ollama.com/download/windows
    ) else (
        echo [*] Waiting for Ollama installer to complete ...
        timeout /t 10 /nobreak >nul
        call :refresh_path
        :: Check known paths again after install
        if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
            set OLLAMA_EXE=%LocalAppData%\Programs\Ollama\ollama.exe
            echo [OK] Ollama installed successfully.
        ) else (
            echo [OK] Ollama installed. ^(PATH will update in your next terminal session^)
        )
    )
)

:: ── 6. Optional image generation (ComfyUI + Pillow) ───────────────────────────
echo.
echo ============================================================
echo   Image Generation (optional)
echo   ComfyUI lets LocalAI generate images locally using
echo   SD, SDXL, Flux, Z-Image, Chroma, and related models.
echo ============================================================
if "!SETUP_ALL_FEATURES!"=="1" (
    set INSTALL_IMG=y
    echo [*] Setup all features: setting up image generation support.
) else (
    set /p INSTALL_IMG="Set up image generation support? [y/N]: "
)
if /i not "!INSTALL_IMG!"=="y" goto :skip_img

:: 6a. Pillow — for in-app image display
echo [*] Installing Pillow (image display) ...
"%PYTHON_EXE%" -m pip install --upgrade --no-warn-conflicts pillow --quiet
if errorlevel 1 (
    echo [WARNING] Pillow install failed. Images will still save but won't show inline.
) else (
    echo [OK] Pillow installed.
)

:: 6b. Locate or install ComfyUI
echo.
set COMFYUI_PATH=
for %%F in ("%~dp0.") do set "APP_ROOT=%%~fF"
set "COMFYUI_DEFAULT=!APP_ROOT!\ComfyUI"

:: Check app-relative default location first
if exist "%COMFYUI_DEFAULT%\main.py" (
    set COMFYUI_PATH=%COMFYUI_DEFAULT%
    echo [OK] Found ComfyUI at: !COMFYUI_PATH!
    goto :comfyui_found
)

:: Check one level up
if exist "%~dp0..\ComfyUI\main.py" (
    for %%F in ("%~dp0..\ComfyUI") do set COMFYUI_PATH=%%~fF
    echo [OK] Found ComfyUI at: !COMFYUI_PATH!
    goto :comfyui_found
)

:: Not found — offer to install it
echo [*] ComfyUI not found in the usual locations.
echo     Default install location: %COMFYUI_DEFAULT%
if "!SETUP_ALL_FEATURES!"=="1" (
    set INSTALL_COMFY=y
    echo [*] Setup all features: downloading and installing ComfyUI.
) else (
    set /p INSTALL_COMFY="Download and install ComfyUI now? [y/N]: "
)
if /i not "!INSTALL_COMFY!"=="y" goto :comfyui_manual

echo [*] Downloading ComfyUI to %COMFYUI_DEFAULT% ...

:: Download as zip from GitHub (no git required) — pinned to a specific release
:: tag so the comfy/ package structure is stable and predictable.
set COMFYUI_TAG=v0.16.4
"%PYTHON_EXE%" -c "import requests,zipfile,io,shutil,pathlib; tag='!COMFYUI_TAG!'; ver=tag.lstrip('v'); root=pathlib.Path(r'!APP_ROOT!'); dst=pathlib.Path(r'!COMFYUI_DEFAULT!'); url=f'https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/{tag}.zip'; print(f'  Downloading ComfyUI {tag}...'); r=requests.get(url,timeout=120); r.raise_for_status(); print('  Extracting...'); z=zipfile.ZipFile(io.BytesIO(r.content)); z.extractall(root); src=root / f'ComfyUI-{ver}'; shutil.move(str(src),str(dst)) if src.exists() else None; print('  Done.')"
if errorlevel 1 (
    echo [ERROR] Download failed. Check internet connection.
    goto :comfyui_skip
)
:: Verify the comfy/ package actually has files — an empty comfy/ breaks startup
if not exist "%COMFYUI_DEFAULT%\comfy\options.py" (
    echo [ERROR] ComfyUI install appears incomplete ^(comfy/options.py missing^).
    echo         Removing partial install and skipping.
    rmdir /s /q "%COMFYUI_DEFAULT%" 2>nul
    goto :comfyui_skip
)
set COMFYUI_PATH=%COMFYUI_DEFAULT%
echo [OK] ComfyUI installed to: !COMFYUI_PATH!
goto :comfyui_found

:comfyui_manual
echo.
set /p COMFYUI_PATH="Enter full path to your ComfyUI folder (or leave blank to skip): "
if "!COMFYUI_PATH!"=="" goto :comfyui_skip
if not exist "!COMFYUI_PATH!\main.py" (
    echo [WARNING] main.py not found at that path — skipping ComfyUI registration.
    set COMFYUI_PATH=
    goto :comfyui_skip
)
echo [OK] Using ComfyUI at: !COMFYUI_PATH!

:comfyui_found
if not defined COMFYUI_PATH goto :comfyui_skip
if exist "!COMFYUI_PATH!\requirements.txt" (
    echo [*] Installing ComfyUI requirements - this may take a few minutes ...
    "%PYTHON_EXE%" -m pip install --no-warn-conflicts -r "!COMFYUI_PATH!\requirements.txt" --quiet
    if errorlevel 1 (
        echo [WARNING] Some ComfyUI requirements failed. Check manually.
    ) else (
        echo [OK] ComfyUI requirements installed.
    )
)
echo [*] Verifying ComfyUI startup dependencies ...
"%PYTHON_EXE%" -m pip install --upgrade --no-input --disable-pip-version-check --no-warn-conflicts SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval "gguf>=0.13.0" "PyYAML>=5.1" "tqdm>=4.27" protobuf "pydantic-settings~=2.0" spandrel "kornia>=0.7.1" PyOpenGL glfw comfyui-frontend-package==1.39.19 comfyui-workflow-templates==0.9.11 comfyui-embedded-docs==0.4.3 --quiet
if errorlevel 1 (
    echo [WARNING] ComfyUI startup dependencies failed. Image generation may fail until the targeted ComfyUI startup packages are installed.
) else (
    echo [OK] ComfyUI startup dependencies verified.
)
echo.
if "!SETUP_ALL_FEATURES!"=="1" (
    if "!SETUP_HAS_GPU!"=="1" (
        set INSTALL_GPU_ACCEL=y
        if "!SETUP_HAS_NVIDIA!"=="1" (
            echo [*] Setup all features: installing NVIDIA CUDA GPU acceleration for !SETUP_NVIDIA_GPU_NAMES!.
        ) else (
            echo [*] Setup all features: installing DirectML GPU acceleration for !SETUP_DML_GPU_NAMES!.
        )
    ) else (
        set INSTALL_GPU_ACCEL=n
        echo [*] Setup all features: skipped GPU image acceleration because no supported local GPU was detected.
    )
) else (
    if "!SETUP_HAS_GPU!"=="1" (
        set /p INSTALL_GPU_ACCEL="Install GPU image acceleration (CUDA for NVIDIA, DirectML for AMD/Intel)? [y/N]: "
    ) else (
        set INSTALL_GPU_ACCEL=n
        echo [*] No supported local GPU detected; skipping GPU image acceleration.
    )
)
if /i "!INSTALL_GPU_ACCEL!"=="y" (
    if "!SETUP_HAS_NVIDIA!"=="1" (
        call :install_cuda_pytorch
        if errorlevel 1 (
            echo [WARNING] CUDA PyTorch setup failed. Image generation can still run on CPU or DirectML if available.
        )
    ) else if "!SETUP_HAS_DML_GPU!"=="1" (
        call :install_directml_pytorch
        if errorlevel 1 (
            echo [WARNING] DirectML packages failed. Image generation can still run on CPU/CUDA if available.
        )
    ) else (
        echo [*] No supported local GPU detected; skipping GPU image acceleration.
    )
)
(echo set LOCALAI_COMFYUI=!COMFYUI_PATH!)> "%~dp0comfyui_path.bat"
:: Update config.json with ComfyUI path
if defined PYTHON_EXE (
    "%PYTHON_EXE%" -c "import json,pathlib; p=pathlib.Path(r'%~dp0config.json'); c=json.loads(p.read_text()) if p.exists() else {}; c['comfyui_dir']=r'!COMFYUI_PATH!'; p.write_text(json.dumps(c,indent=2))" 2>nul
)
echo [OK] ComfyUI path saved.
echo.
echo  ** Next step: download a model checkpoint **
echo     Open LocalAI ^> Models tab ^> filter "Image Generation"
echo     Click "Get Model" to download and install automatically.
goto :img_done

:comfyui_skip
echo [*] Skipped ComfyUI. Re-run setup.bat any time to add it.
goto :img_done

:skip_img
echo [*] Skipped image generation. Re-run setup.bat any time to add it.

:img_done

:: ── 6d. Verify dependency consistency ────────────────────────────────────────
echo.
echo [*] Verifying Python dependency consistency ...
"%PYTHON_EXE%" -m pip check
if errorlevel 1 (
    echo [WARNING] pip detected dependency conflicts above. LocalAI may crash on launch.
    echo           Most common cause: huggingface-hub 1.x conflicting with transformers ^<1.0 pin.
    echo           Re-run setup.bat and choose 'y' to all optional installs, or run:
    echo             "%PYTHON_EXE%" -m pip install --upgrade "huggingface-hub^>=0.34.0,^<1.0"
) else (
    echo [OK] No dependency conflicts detected.
)

:: ── 7. Final smoke test ───────────────────────────────────────────────────────
echo.
echo [*] Verifying installation ...
"%PYTHON_EXE%" -c "import customtkinter, requests, psutil; print('  imports OK')" 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Smoke test failed — something is not right with the install.
    echo           Try closing this window and running setup.bat again.
    echo           If it keeps failing, check your internet connection and antivirus.
    pause
    exit /b 1
)

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Launch the app:  double-click run.bat
echo ============================================================
echo.
pause
exit /b 0


:: ════════════════════════════════════════════════════════════════════════════
:: Subroutines
:: ════════════════════════════════════════════════════════════════════════════

:detect_setup_hardware
set SETUP_HAS_GPU=0
set SETUP_HAS_NVIDIA=0
set SETUP_HAS_DML_GPU=0
set SETUP_GPU_NAMES=
set SETUP_NVIDIA_GPU_NAMES=
set SETUP_DML_GPU_NAMES=
for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "$g=Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -and $_.Name -notmatch 'Microsoft|Basic|Remote|Hyper-V' -and $_.Name -match 'NVIDIA|GeForce|RTX|Quadro|AMD|Radeon|Intel|Arc|Iris|UHD|Xe' }; if ($g) { ($g | ForEach-Object { $_.Name }) -join ', ' }" 2^>nul`) do set "SETUP_GPU_NAMES=%%G"
if defined SETUP_GPU_NAMES set SETUP_HAS_GPU=1
for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "$g=Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -and $_.Name -notmatch 'Microsoft|Basic|Remote|Hyper-V' -and $_.Name -match 'NVIDIA|GeForce|RTX|Quadro' }; if ($g) { ($g | ForEach-Object { $_.Name }) -join ', ' }" 2^>nul`) do set "SETUP_NVIDIA_GPU_NAMES=%%G"
if defined SETUP_NVIDIA_GPU_NAMES set SETUP_HAS_NVIDIA=1
for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "$g=Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -and $_.Name -notmatch 'Microsoft|Basic|Remote|Hyper-V|NVIDIA|GeForce|RTX|Quadro' -and $_.Name -match 'AMD|Radeon|Intel|Arc|Iris|UHD|Xe' }; if ($g) { ($g | ForEach-Object { $_.Name }) -join ', ' }" 2^>nul`) do set "SETUP_DML_GPU_NAMES=%%G"
if defined SETUP_DML_GPU_NAMES set SETUP_HAS_DML_GPU=1
:: v2026.06.01.7 (Ron, 2026-06-01): pick exactly ONE onnxruntime variant
:: based on detected GPU vendor. onnxruntime / onnxruntime-gpu /
:: onnxruntime-directml all install into the same onnxruntime/ namespace
:: and are mutually exclusive. Installing two side-by-side leaves the
:: namespace half-broken (onnxruntime.__file__ is None, InferenceSession
:: vanishes) and Toolbox features that do `from onnxruntime import
:: InferenceSession` crash. Branching here so every later pip install
:: step uses the same chosen variant.
if "!SETUP_HAS_NVIDIA!"=="1" (
    set "SETUP_ONNX_PKG=onnxruntime-gpu"
    set "SETUP_ONNX_GENAI_PKG=onnxruntime-genai-cuda"
    set "SETUP_ONNX_EP=CUDAExecutionProvider"
    set "SETUP_ONNX_LABEL=NVIDIA CUDA"
) else if "!SETUP_HAS_DML_GPU!"=="1" (
    set "SETUP_ONNX_PKG=onnxruntime-directml"
    set "SETUP_ONNX_GENAI_PKG=onnxruntime-genai-directml"
    set "SETUP_ONNX_EP=DmlExecutionProvider"
    set "SETUP_ONNX_LABEL=DirectML"
) else (
    set "SETUP_ONNX_PKG=onnxruntime"
    set "SETUP_ONNX_GENAI_PKG=onnxruntime-genai"
    set "SETUP_ONNX_EP=CPUExecutionProvider"
    set "SETUP_ONNX_LABEL=CPU"
)
goto :eof

:install_cuda_pytorch
echo [*] Checking NVIDIA CUDA PyTorch ...
"%PYTHON_EXE%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    "%PYTHON_EXE%" -c "import torch; print(f'  CUDA already available: {torch.__version__} / CUDA {torch.version.cuda} / {torch.cuda.get_device_name(0)}')" 2>&1
    echo [OK] NVIDIA CUDA PyTorch already verified.
    exit /b 0
)
echo [*] Installing NVIDIA CUDA PyTorch image packages ...
echo     This downloads large CUDA wheels and can take several minutes.
echo     Progress will be shown below; do not close this window.
echo     Note: the NVIDIA cu128 channel currently ships PyTorch 2.11.0;
echo     if a newer PyTorch is already installed from another channel
echo     ^(e.g. torch-directml 2.12.0^) pip will replace it with the cu128
echo     wheel. That replacement is expected and required for CUDA.
"%PYTHON_EXE%" -m pip uninstall -y torch torchvision torchaudio torch-directml
echo.
:: --no-cache-dir was needed pre-v5.3.10 because pip cachecontrol spooled the
:: 2.8 GB CUDA wheel into a Windows %TEMP% temp file that could fail with
:: OSError 28 even on a half-empty disk (commit-limit / Defender lock /
:: roaming profile container cap). v5.3.10 redirects PIP_CACHE_DIR / TMP / TEMP /
:: HF_HOME / TORCH_HOME to <app>\.cache\ BEFORE this script runs anything,
:: so the spool now lives on the install drive and the workaround can be
:: removed. We keep --disable-pip-version-check for terse output and
:: --no-input so unattended runs never hang on a pip prompt.
"%PYTHON_EXE%" -m pip install --upgrade --no-input --disable-pip-version-check --no-warn-conflicts torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERROR] CUDA PyTorch install failed.
    echo         The pip cache and TMP have been redirected to %~dp0.cache\ so
    echo         OSError 28 on a small profile drive should no longer fire.
    echo         If you still saw "No space left on device", check the free space
    echo         on the drive holding %~dp0.cache\ and that nothing ^(antivirus,
    echo         a profile virtualization policy^) is locking files in that folder.
    echo         Attempting to recover any partial pip cache ...
    "%PYTHON_EXE%" -m pip cache purge >nul 2>&1
    exit /b 1
)
echo [*] Verifying NVIDIA CUDA PyTorch ...
"%PYTHON_EXE%" -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA version: {torch.version.cuda}'); assert torch.cuda.is_available(), 'CUDA is not available after install'; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>&1
if errorlevel 1 (
    echo [ERROR] CUDA PyTorch installed but CUDA did not initialize.
    echo         Check the NVIDIA driver version or run fix_nvidia_pytorch.bat for details.
    exit /b 1
)
echo [OK] NVIDIA CUDA PyTorch verified.
exit /b 0

:install_directml_pytorch
:: v5.5.9 (Ron, 2026-05-26): bail early on Windows ARM64. torch-directml
:: has no Windows-ARM64 wheel on PyPI, so an install attempt will fail and
:: leave the user with a half-broken environment. Chat / vision / ONNX
:: paths still work on CPU; image generation surfaces an "unsupported on
:: Snapdragon X" panel in the UI (src/app.py::_build_image_page).
"%PYTHON_EXE%" -c "import platform, sys; sys.exit(0 if platform.machine().upper() == 'ARM64' else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Windows ARM64 / Snapdragon X detected.
    echo        Skipping DirectML PyTorch install ^(no ARM64 wheel on PyPI^).
    echo        Image generation will show an "unsupported" panel in the UI.
    echo        Chat, vision, ONNX and embeddings remain available on CPU.
    exit /b 0
)
:: v5.5.10 (Ron, 2026-05-26): self-healing pre-check. If torch + torchaudio
:: + torch_directml already import cleanly AND torch_directml.is_available()
:: is True, the existing install is ABI-coherent - skip the reinstall.
:: If the import fails (the "Entry Point Not Found: torch_library_impl in
:: _torchaudio.pyd" Windows dialog scenario), torch and torchaudio have
:: drifted out of ABI sync. SetErrorMode^(0x8001 = SEM_FAILCRITICALERRORS ^|
:: SEM_NOOPENFILEERRORBOX^) suppresses the OS dialog so the failure surfaces
:: as a clean nonzero exit instead of blocking on a confusing popup. On
:: drift we uninstall before reinstalling: --upgrade alone cannot repair
:: the drift because torch-directml's torch pin keeps pip from touching
:: torch while torchaudio gets bumped independently, leaving them mismatched.
echo [*] Checking DirectML PyTorch ABI health ...
"%PYTHON_EXE%" -c "import ctypes, sys; ctypes.windll.kernel32.SetErrorMode(0x8001); import torch, torchaudio, torch_directml; sys.exit(0 if torch_directml.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    "%PYTHON_EXE%" -c "import ctypes; ctypes.windll.kernel32.SetErrorMode(0x8001); import torch, torchaudio, torch_directml; print(f'  Already healthy: torch {torch.__version__} / torchaudio {torchaudio.__version__} / DirectML {torch_directml.device_name(0)}')" 2>&1
    echo [OK] DirectML PyTorch already verified.
    exit /b 0
)
echo [*] DirectML PyTorch missing or torch/torchaudio drifted out of ABI sync.
echo     Performing clean reinstall ^(uninstall first, then matched-set install^) ...
"%PYTHON_EXE%" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>&1
:: v5.5.11 (Ron, 2026-05-26): two-step install to prevent torchaudio drift.
:: v5.5.10 used `pip install --upgrade torch-directml torch torchvision torchaudio`
:: which let pip's resolver pick the LATEST torchaudio (e.g. 2.11.0) against
:: torch-directml's pinned torch (2.4.1), producing WinError 127 /
:: "torch_library_impl in _torchaudio.pyd" on Intel AI PC. Step 1 installs
:: torch-directml ALONE so its torch pin lands first. Step 2 discovers that
:: pinned torch version at runtime and installs torchaudio==<same-version> +
:: torchvision against it. Future-proof: when MS bumps torch-directml to a new
:: torch, this auto-adapts without a code change.
:: Cache redirection happens at script top via PIP_CACHE_DIR / TMP / TEMP
:: pointing at %~dp0.cache so the multi-GB DirectML+torch spool lands on the
:: install drive instead of the (possibly tiny) profile drive.
echo [*] Installing torch-directml ^(pins torch^) ...
"%PYTHON_EXE%" -m pip install --upgrade --no-input --disable-pip-version-check --no-warn-conflicts torch-directml
if errorlevel 1 (
    echo [ERROR] torch-directml install failed.
    echo         The pip cache + TMP have been redirected to %~dp0.cache\.
    echo         If you still saw OSError 28, free space on the install drive or
    echo         check that nothing is locking files under %~dp0.cache\.
    "%PYTHON_EXE%" -m pip cache purge >nul 2>&1
    exit /b 1
)
echo [*] Discovering pinned torch version ...
set "VFILE=%TEMP%\__localai_torchver_setup.txt"
set "TORCH_VER="
"%PYTHON_EXE%" -c "import torch; print(torch.__version__.split(chr(43))[0])" > "%VFILE%" 2>nul
if exist "%VFILE%" set /p TORCH_VER=<"%VFILE%"
if exist "%VFILE%" del "%VFILE%" >nul 2>&1
if not defined TORCH_VER (
    echo [ERROR] Could not discover installed torch version after torch-directml install.
    echo         Run fix_directml_pytorch.bat for verbose diagnostics.
    exit /b 1
)
echo [*] torch pinned to %TORCH_VER%; installing matched torchaudio + torchvision ...
"%PYTHON_EXE%" -m pip install --no-input --disable-pip-version-check --no-warn-conflicts "torchaudio==%TORCH_VER%" torchvision
if errorlevel 1 (
    echo [ERROR] Matched torchaudio==%TORCH_VER% / torchvision install failed.
    echo         If image generation now fails with "Entry Point Not Found:
    echo         torch_library_impl", torch and torchaudio have drifted out of
    echo         ABI sync ^- run fix_directml_pytorch.bat to realign them.
    "%PYTHON_EXE%" -m pip cache purge >nul 2>&1
    exit /b 1
)
:: v5.5.10: post-install ABI verify with SetErrorMode-suppressed import. If
:: this still fails after a clean uninstall+reinstall, point the user at
:: fix_directml_pytorch.bat for verbose diagnostics (likely a GPU driver
:: issue or a corrupted Python environment that needs manual attention).
echo [*] Verifying DirectML PyTorch ABI alignment ...
"%PYTHON_EXE%" -c "import ctypes; ctypes.windll.kernel32.SetErrorMode(0x8001); import torch, torchaudio, torch_directml; assert torch_directml.is_available(), 'torch-directml installed but no DirectML device is visible'; print(f'  PyTorch: {torch.__version__}'); print(f'  torchaudio: {torchaudio.__version__}'); print(f'  DirectML device: {torch_directml.device_name(0)}')" 2>&1
if errorlevel 1 (
    echo [WARNING] DirectML PyTorch ABI verify failed even after clean reinstall.
    echo           Run fix_directml_pytorch.bat manually for verbose diagnostics.
    exit /b 1
)
echo [OK] DirectML acceleration packages installed and verified.
exit /b 0

:find_python
:: Search PATH first (skips the Store python.exe stub — it returns exit code 9009)
for %%P in (python.exe python3.exe py.exe) do (
    "%%~$PATH:P" --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_EXE=%%~$PATH:P
        goto :eof
    )
)
:: Search common install locations for any Python 3.x
for /d %%D in (
    "%LocalAppData%\Programs\Python\Python3*"
    "%ProgramFiles%\Python3*"
    "%ProgramFiles(x86)%\Python3*"
) do (
    if exist "%%D\python.exe" (
        set PYTHON_EXE=%%D\python.exe
        goto :eof
    )
)
:: Check the Python Launcher which can resolve to any installed version
if exist "%LocalAppData%\Programs\Python\Launcher\py.exe" (
    "%LocalAppData%\Programs\Python\Launcher\py.exe" --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_EXE=%LocalAppData%\Programs\Python\Launcher\py.exe
        goto :eof
    )
)
goto :eof

:refresh_path
:: Read the updated Machine + User PATH from the registry and apply to this session.
:: This is needed because winget updates the registry but not the current cmd session.
for /f "usebackq delims=" %%M in (
    `powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('PATH','Machine')"`
) do set "MACHINE_PATH=%%M"
for /f "usebackq delims=" %%U in (
    `powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('PATH','User')"`
) do set "USER_PATH=%%U"
if defined MACHINE_PATH if defined USER_PATH set "PATH=!MACHINE_PATH!;!USER_PATH!"
goto :eof

:diagnose_msi_service
:: Check Windows Installer service health and provide troubleshooting guidance
echo Diagnosing Windows Installer service...
echo.

:: Check if msiexec processes are running
set MSI_RUNNING=0
for /f %%i in ('tasklist /FI "IMAGENAME eq msiexec.exe" 2^>nul ^| find /c /i "msiexec.exe"') do set MSI_COUNT=%%i
if %MSI_COUNT% gtr 0 (
    echo Found %MSI_COUNT% msiexec.exe process^(es^) running:
    tasklist /FI "IMAGENAME eq msiexec.exe" /FO TABLE 2>nul | findstr /V "Image"
    set MSI_RUNNING=1
    echo.
)

:: Check Windows Installer service status
for /f "tokens=3" %%s in ('sc query msiserver ^| findstr /C:"STATE"') do set MSI_STATE=%%s

echo Windows Installer Service ^(msiserver^): !MSI_STATE!
echo.

:: Provide specific guidance based on what we found
if !MSI_RUNNING! equ 1 (
    echo [*] Active installations detected. This could be:
    echo     - A legitimate installation in progress ^(wait for it to finish^)
    echo     - A stuck/hung installer that needs to be terminated
    echo.
    echo [*] Recommended actions:
    echo     1. Check if any installation windows are open and complete them
    echo     2. If no installation UI is visible, an installer may be stuck
    echo.
    echo [*] Do not kill msiexec.exe from this setup script; it may belong to another installer.
    echo [*] Finish the other installation or reboot Windows, then re-run setup.bat.
) else if "!MSI_STATE!"=="STOPPED" (
    echo [*] The Windows Installer service is stopped but may be locked.
    echo.
    set /p START_MSI="Attempt to start the Windows Installer service? [y/N]: "
    if /i "!START_MSI!"=="y" (
        echo [*] Starting Windows Installer service...
        net start msiserver >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Failed to start service. You may need Administrator rights.
            echo         Try running this script as Administrator, or start the service manually.
        ) else (
            echo [OK] Service started. Please re-run setup.bat now.
        )
    ) else (
        echo [*] Run as Administrator and use: net start msiserver
    )
) else (
    echo [*] The service appears running but may be locked or in a bad state.
    echo.
    echo [*] Recommended actions:
    echo     1. Restart your computer ^(most reliable^)
    echo     2. Or manually restart the Windows Installer service:
    echo        - Open Services ^(services.msc^)
    echo        - Find "Windows Installer"
    echo        - Right-click ^> Restart
    echo     3. Re-run this script
    echo.
    set /p RESTART_PC="Restart computer now? [y/N]: "
    if /i "!RESTART_PC!"=="y" (
        echo [*] Restarting in 30 seconds... Press Ctrl+C to cancel.
        shutdown /r /t 30 /c "Restarting to fix Windows Installer service"
    )
)
echo.
goto :eof
