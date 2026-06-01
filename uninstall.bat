@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: Detect /silent flag (unattended mode -- all prompts answered with safe defaults, nothing deleted)
set SILENT=0
if /i "%~1"=="/silent" set SILENT=1
if /i "%~1"=="/y" set SILENT=1

echo ============================================================
echo   LocalAI Studio -- Uninstaller
echo   Restore your system to a clean state
echo ============================================================
echo.
echo This script will walk you through removing each component
echo installed by setup.bat. You choose what to remove.
echo.
echo Press Ctrl+C at any time to abort.
echo.
if !SILENT! equ 0 pause

:: ── Track totals ────────────────────────────────────────────────────────────
set TOTAL_REMOVED=0

:: ════════════════════════════════════════════════════════════════════════════
:: 1. Stop running processes
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 1 of 8: Stop Running Processes
echo ============================================================
echo.
set PROCS_STOPPED=0

call :handle_ollama

:: Check for ComfyUI (python running main.py)
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /i "python.exe" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   [^^!] Python processes detected ^(may include ComfyUI^).
    echo       If LocalAI or ComfyUI is running, close the app first.
    if !SILENT! equ 1 (set CONTINUE_PYTHON=) else set /p CONTINUE_PYTHON="      Continue anyway? [Y/n]: "
    if /i "!CONTINUE_PYTHON!"=="n" (
        echo   [*] Please close LocalAI and ComfyUI, then re-run this script.
        if !SILENT! equ 0 pause
        exit /b 0
    )
)

:: ════════════════════════════════════════════════════════════════════════════
:: 2. Ollama downloaded models
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 2 of 8: Ollama Downloaded Models
echo   Location: %USERPROFILE%\.ollama\models
echo ============================================================
echo.

if exist "%USERPROFILE%\.ollama\models" (
    :: Estimate size
    set OLLAMA_SIZE=unknown
    for /f "tokens=3" %%s in ('dir /s "%USERPROFILE%\.ollama\models" 2^>nul ^| findstr /C:"bytes" ^| findstr /V /C:"free"') do set OLLAMA_SIZE=%%s
    echo   [^^!] Found Ollama model cache.
    echo       Estimated size: !OLLAMA_SIZE! bytes
    echo.
    echo       This removes ALL Ollama models ^(used by any app, not just LocalAI^).
    echo.
    if !SILENT! equ 1 (set DEL_OLLAMA_MODELS=) else set /p DEL_OLLAMA_MODELS="      Delete all Ollama models? [y/N]: "
    if /i "!DEL_OLLAMA_MODELS!"=="y" (
        echo   [*] Deleting Ollama model cache ...
        rmdir /s /q "%USERPROFILE%\.ollama\models" 2>nul
        if not exist "%USERPROFILE%\.ollama\models" (
            echo   [OK] Ollama models deleted.
            set /a TOTAL_REMOVED+=1
        ) else (
            echo   [WARNING] Could not fully delete. Some files may be locked.
            echo            Close Ollama and try again, or delete manually:
            echo            %USERPROFILE%\.ollama\models
        )
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [OK] No Ollama model cache found.
)

:: ── App-local Ollama models directory (v5.3.7+ relocation target) ──────────
:: When the user accepted "Move Ollama models directory" in Settings →
:: Storage, OLLAMA_MODELS was redirected to <app>\Ollama and (in some
:: setups) the daemon's blobs were copied there.  Without this block the
:: uninstaller would leave that copy AND a USER-scope OLLAMA_MODELS env
:: var pointing at a path that no longer exists.
if exist "%~dp0Ollama\models" (
    echo.
    echo   [^^!] Found app-local Ollama models directory: %~dp0Ollama
    echo       This is the directory LocalAI relocated Ollama to.
    if !SILENT! equ 1 (set DEL_APP_OLLAMA=) else set /p DEL_APP_OLLAMA="      Delete app-local Ollama models? [y/N]: "
    if /i "!DEL_APP_OLLAMA!"=="y" (
        rmdir /s /q "%~dp0Ollama" 2>nul
        if not exist "%~dp0Ollama" (
            echo   [OK] App-local Ollama models deleted.
            set /a TOTAL_REMOVED+=1
        ) else (
            echo   [WARNING] Could not fully delete %~dp0Ollama.
            echo            Close Ollama and any LLM front-end, then retry.
        )
    ) else (
        echo   [*] Skipped.
    )
)

:: ── Clear OLLAMA_MODELS USER env var if it points at an app-local path ────
:: Leaving the env var dangling causes the Ollama daemon and every Ollama-
:: speaking app (Open-WebUI, Continue.dev, …) to fail with "models
:: directory not found" on next launch.
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v OLLAMA_MODELS 2^>nul ^| findstr /I "OLLAMA_MODELS"') do set CURRENT_OLLAMA_MODELS=%%B
if defined CURRENT_OLLAMA_MODELS (
    set CLEAR_OLLAMA_ENV=
    :: If it points anywhere under the app folder OR to a non-existent
    :: path, offer to clear it.
    if not exist "!CURRENT_OLLAMA_MODELS!" set CLEAR_OLLAMA_ENV=1
    echo "!CURRENT_OLLAMA_MODELS!" | findstr /I /B /C:"\"%~dp0" >nul && set CLEAR_OLLAMA_ENV=1
    if defined CLEAR_OLLAMA_ENV (
        echo.
        echo   [^^!] OLLAMA_MODELS env var is set to:
        echo       !CURRENT_OLLAMA_MODELS!
        echo       ^(now stale — points at LocalAI's app folder or a missing path^)
        if !SILENT! equ 1 (set CLEAR_OLLAMA=) else set /p CLEAR_OLLAMA="      Clear this USER env var? [Y/n]: "
        if /i not "!CLEAR_OLLAMA!"=="n" (
            reg delete "HKCU\Environment" /v OLLAMA_MODELS /f >nul 2>&1
            if not errorlevel 1 (
                echo   [OK] OLLAMA_MODELS cleared. Restart your shell or sign out/in to apply.
                set /a TOTAL_REMOVED+=1
            ) else (
                echo   [WARNING] Could not clear OLLAMA_MODELS. Clear it manually in System Properties → Environment Variables.
            )
        ) else (
            echo   [*] Skipped.
        )
    )
)

:: ════════════════════════════════════════════════════════════════════════════
:: 3. ComfyUI + image generation models
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 3 of 8: ComfyUI ^& Image Generation Models
echo ============================================================
echo.

:: Priority 1 — standard location (%LocalAppData%\LocalAI\ComfyUI)
set COMFYUI_FOUND=
if exist "%LocalAppData%\LocalAI\ComfyUI\main.py" set COMFYUI_FOUND=%LocalAppData%\LocalAI\ComfyUI

:: Priority 2 — legacy embedded location (inside app folder)
if not defined COMFYUI_FOUND (
    if exist "%~dp0ComfyUI\main.py" set COMFYUI_FOUND=%~dp0ComfyUI
)

:: Priority 3 — path stored in config.json (catches installs in custom or older locations)
if exist "%~dp0config.json" (
    for /f "usebackq delims=" %%P in (
        `powershell -NoProfile -Command "try{(Get-Content '%~dp0config.json'^|ConvertFrom-Json).comfyui_dir}catch{}" 2^>nul`
    ) do set CFG_COMFYUI=%%P
)
if defined CFG_COMFYUI (
    if exist "!CFG_COMFYUI!\main.py" (
        if /i not "!CFG_COMFYUI!"=="%LocalAppData%\LocalAI\ComfyUI" (
            if /i not "!CFG_COMFYUI!"=="%~dp0ComfyUI" (
                :: Non-standard location recorded in config
                if not defined COMFYUI_FOUND (
                    set COMFYUI_FOUND=!CFG_COMFYUI!
                ) else (
                    echo   [^^!] config.json also references a separate ComfyUI location:
                    echo       !CFG_COMFYUI!
                    if !SILENT! equ 1 (set DEL_CFG_COMFYUI=) else set /p DEL_CFG_COMFYUI="      Delete this location too? [y/N]: "
                    if /i "!DEL_CFG_COMFYUI!"=="y" (
                        rmdir /s /q "!CFG_COMFYUI!" 2>nul
                        echo   [OK] Additional ComfyUI location deleted.
                        set /a TOTAL_REMOVED+=1
                    ) else (
                        echo   [*] Skipped.
                    )
                )
            )
        )
    )
)

if defined COMFYUI_FOUND (
    echo   [^^!] Found ComfyUI at: !COMFYUI_FOUND!
    echo.
    echo       This includes ComfyUI itself and any downloaded image models
    echo       ^(Stable Diffusion, SDXL, Flux checkpoints, CLIP encoders, VAE^).
    echo.
    if !SILENT! equ 1 (set DEL_COMFYUI=) else set /p DEL_COMFYUI="      Delete ComfyUI and all image models? [y/N]: "
    if /i "!DEL_COMFYUI!"=="y" (
        echo   [*] Deleting ComfyUI ...
        rmdir /s /q "!COMFYUI_FOUND!" 2>nul
        if not exist "!COMFYUI_FOUND!" (
            echo   [OK] ComfyUI deleted.
            set /a TOTAL_REMOVED+=1
        ) else (
            echo   [WARNING] Could not fully delete. Try closing all apps and delete manually:
            echo            !COMFYUI_FOUND!
        )
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [OK] No ComfyUI installation found.
)

:: ════════════════════════════════════════════════════════════════════════════
:: 4. ONNX / HuggingFace downloaded models
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 4 of 8: ONNX / HuggingFace Model Cache
echo ============================================================
echo.

:: Location A — new default (%LocalAppData%\LocalAI\onnx and \ov)
set ONNX_NEW_FOUND=
if exist "%LocalAppData%\LocalAI\onnx" set ONNX_NEW_FOUND=1
if exist "%LocalAppData%\LocalAI\ov"   set ONNX_NEW_FOUND=1

if defined ONNX_NEW_FOUND (
    echo   [^^!] Found LocalAI ONNX/OpenVINO models in: %LocalAppData%\LocalAI
    echo       ^(onnx\ and/or ov\ subdirectories^)
    if !SILENT! equ 1 (set DEL_ONNX_NEW=) else set /p DEL_ONNX_NEW="      Delete these model files? [y/N]: "
    if /i "!DEL_ONNX_NEW!"=="y" (
        if exist "%LocalAppData%\LocalAI\onnx" rmdir /s /q "%LocalAppData%\LocalAI\onnx" 2>nul
        if exist "%LocalAppData%\LocalAI\ov"   rmdir /s /q "%LocalAppData%\LocalAI\ov"   2>nul
        echo   [OK] ONNX/OpenVINO models deleted.
        set /a TOTAL_REMOVED+=1
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [OK] No ONNX/OpenVINO models found in %LocalAppData%\LocalAI.
)

:: Location B — legacy app-relative models folder
echo.
if exist "%~dp0models" (
    echo   [^^!] Found legacy ONNX models folder: %~dp0models
    if !SILENT! equ 1 (set DEL_ONNX_OLD=) else set /p DEL_ONNX_OLD="      Delete legacy models folder? [y/N]: "
    if /i "!DEL_ONNX_OLD!"=="y" (
        rmdir /s /q "%~dp0models" 2>nul
        echo   [OK] Legacy models folder deleted.
        set /a TOTAL_REMOVED+=1
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [OK] No legacy app-relative models folder found.
)

:: Location C — custom path from config.json (if different from A and B)
if exist "%~dp0config.json" (
    for /f "usebackq delims=" %%P in (
        `powershell -NoProfile -Command "try{(Get-Content '%~dp0config.json'^|ConvertFrom-Json).models_dir}catch{}" 2^>nul`
    ) do set CFG_MODELS=%%P
)
if defined CFG_MODELS (
    if /i not "!CFG_MODELS!"=="%LocalAppData%\LocalAI" (
        if /i not "!CFG_MODELS!"=="%~dp0models" (
            if exist "!CFG_MODELS!" (
                echo.
                echo   [^^!] config.json references a non-standard models location:
                echo       !CFG_MODELS!
                if !SILENT! equ 1 (set DEL_CFG_MODELS=) else set /p DEL_CFG_MODELS="      Delete this models folder? [y/N]: "
                if /i "!DEL_CFG_MODELS!"=="y" (
                    rmdir /s /q "!CFG_MODELS!" 2>nul
                    echo   [OK] Custom models folder deleted.
                    set /a TOTAL_REMOVED+=1
                ) else (
                    echo   [*] Skipped.
                )
            )
        )
    )
)

:: Clean up %LocalAppData%\LocalAI if now empty (ComfyUI + models both gone)
if exist "%LocalAppData%\LocalAI" (
    dir /b "%LocalAppData%\LocalAI" 2>nul | findstr "." >nul 2>&1
    if errorlevel 1 (
        rmdir "%LocalAppData%\LocalAI" 2>nul
        echo   [OK] Removed empty %LocalAppData%\LocalAI folder.
    )
)

:: HuggingFace cache (shared by many tools)
echo.
if exist "%USERPROFILE%\.cache\huggingface" (
    echo   [^^!] Found HuggingFace cache: %USERPROFILE%\.cache\huggingface
    echo       WARNING: This cache is shared by all HuggingFace tools on your system.
    echo       Only delete if you are sure no other apps use it.
    echo.
    if !SILENT! equ 1 (set DEL_HF_CACHE=) else set /p DEL_HF_CACHE="      Delete HuggingFace cache? [y/N]: "
    if /i "!DEL_HF_CACHE!"=="y" (
        rmdir /s /q "%USERPROFILE%\.cache\huggingface" 2>nul
        echo   [OK] HuggingFace cache deleted.
        set /a TOTAL_REMOVED+=1
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [OK] No HuggingFace cache found.
)

:: App-local HuggingFace cache (v5.3.10+ canonical location)
:: setup.bat sets HF_HOME=%~dp0.cache\huggingface so downloads land next
:: to the app instead of the user profile.  This block catches that copy
:: even if the user has never had the legacy ~\.cache\huggingface dir.
if exist "%~dp0.cache\huggingface" (
    echo.
    echo   [^^!] Found app-local HuggingFace cache: %~dp0.cache\huggingface
    echo       This is LocalAI's own download cache ^(toolbox models, HF Hub^).
    echo       Safe to delete — only LocalAI writes here.
    if !SILENT! equ 1 (set DEL_APP_HF=) else set /p DEL_APP_HF="      Delete app-local HuggingFace cache? [y/N]: "
    if /i "!DEL_APP_HF!"=="y" (
        rmdir /s /q "%~dp0.cache\huggingface" 2>nul
        if not exist "%~dp0.cache\huggingface" (
            echo   [OK] App-local HuggingFace cache deleted.
            set /a TOTAL_REMOVED+=1
        ) else (
            echo   [WARNING] Could not fully delete. Close LocalAI and retry.
        )
    ) else (
        echo   [*] Skipped.
    )
)

:: ════════════════════════════════════════════════════════════════════════════
:: 5. Python pip packages installed by setup.bat
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 5 of 8: Python Packages
echo ============================================================
echo.

:: Find Python
set PYTHON_EXE=
if exist "%~dp0python_path.bat" (
    call "%~dp0python_path.bat"
    if defined LOCALAI_PYTHON if exist "%LOCALAI_PYTHON%" set PYTHON_EXE=%LOCALAI_PYTHON%
)
if not defined PYTHON_EXE (
    for %%P in (python.exe python3.exe) do (
        "%%~$PATH:P" --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_EXE=%%~$PATH:P
            goto :found_pip_python
        )
    )
)
:found_pip_python

if defined PYTHON_EXE (
    echo   [*] Using Python: %PYTHON_EXE%
    echo.
    echo   The following packages were installed by setup.bat:
    echo.
    echo     Core:     customtkinter, requests, psutil
    echo     ONNX:     onnxruntime-directml, onnxruntime-genai-directml, optimum, transformers, huggingface-hub
    echo     OpenVINO: openvino, openvino-genai
    echo     Utility:  torch, transformers, sentence-transformers, diffusers, pyttsx3, piper-tts, timm, librosa, hf_xet, etc.
    echo     Image:    pillow, torch, torchvision, torchaudio, torch-directml
    echo     ComfyUI:  aiohttp, einops, safetensors, gguf, PyYAML, tqdm, SQLAlchemy, etc.
    echo.
    echo   a^) Remove ALL packages listed above
    echo   b^) Remove only ONNX/DirectML packages
    echo   c^) Remove only OpenVINO packages
    echo   d^) Remove only utility/demo packages
    echo   e^) Remove only image generation packages ^(torch, pillow, ComfyUI deps^)
    echo   f^) Skip -- keep all packages
    echo.
    if !SILENT! equ 1 (set PIP_CHOICE=) else set /p PIP_CHOICE="      Choose [a/b/c/d/e/f]: "

    if /i "!PIP_CHOICE!"=="a" (
        echo   [*] Removing all LocalAI-related packages ...
        "%PYTHON_EXE%" -m pip uninstall -y customtkinter requests psutil 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y onnxruntime-directml onnxruntime-genai-directml onnxruntime optimum transformers huggingface-hub tokenizers 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y openvino openvino-genai 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y accelerate sentence-transformers diffusers soundfile scipy pyttsx3 einops timm sentencepiece peft librosa backoff hf_xet 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y pillow torch torchvision torchaudio torch-directml 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y aiohttp einops safetensors scipy 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y gguf PyYAML tqdm SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval protobuf pydantic-settings spandrel kornia PyOpenGL glfw comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs 2>nul
        echo   [OK] Packages removed.
        set /a TOTAL_REMOVED+=1
    ) else if /i "!PIP_CHOICE!"=="b" (
        echo   [*] Removing ONNX/DirectML packages ...
        "%PYTHON_EXE%" -m pip uninstall -y onnxruntime-directml onnxruntime-genai-directml onnxruntime optimum transformers huggingface-hub tokenizers 2>nul
        echo   [OK] ONNX packages removed.
        set /a TOTAL_REMOVED+=1
    ) else if /i "!PIP_CHOICE!"=="c" (
        echo   [*] Removing OpenVINO packages ...
        "%PYTHON_EXE%" -m pip uninstall -y openvino openvino-genai 2>nul
        echo   [OK] OpenVINO packages removed.
        set /a TOTAL_REMOVED+=1
    ) else if /i "!PIP_CHOICE!"=="d" (
        echo   [*] Removing utility/demo packages ...
        "%PYTHON_EXE%" -m pip uninstall -y torch torchvision transformers accelerate safetensors sentence-transformers soundfile scipy pillow timm sentencepiece librosa backoff onnxruntime optimum diffusers pyttsx3 piper-tts einops peft huggingface-hub hf_xet tokenizers 2>nul
        echo   [OK] Utility packages removed.
        set /a TOTAL_REMOVED+=1
    ) else if /i "!PIP_CHOICE!"=="e" (
        echo   [*] Removing image generation packages ...
        "%PYTHON_EXE%" -m pip uninstall -y pillow torch torchvision torchaudio torch-directml 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y aiohttp einops safetensors scipy 2>nul
        "%PYTHON_EXE%" -m pip uninstall -y gguf PyYAML tqdm SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval protobuf pydantic-settings spandrel kornia PyOpenGL glfw comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs 2>nul
        echo   [OK] Image generation packages removed.
        set /a TOTAL_REMOVED+=1
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [WARNING] Python not found. Cannot remove pip packages automatically.
    echo            You can remove them manually later with:
    echo            pip uninstall customtkinter requests psutil onnxruntime-directml ...
)

:: ════════════════════════════════════════════════════════════════════════════
:: 6. Uninstall Ollama
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 6 of 8: Uninstall Ollama
echo ============================================================
echo.

set OLLAMA_INSTALLED=
if exist "%LocalAppData%\Programs\Ollama\ollama.exe" set OLLAMA_INSTALLED=1
if exist "%ProgramFiles%\Ollama\ollama.exe" set OLLAMA_INSTALLED=1

if defined OLLAMA_INSTALLED (
    echo   [^^!] Ollama is installed on this system.
    echo       NOTE: If other apps use Ollama, do NOT uninstall it.
    echo.
    if !SILENT! equ 1 (set DEL_OLLAMA=) else set /p DEL_OLLAMA="      Uninstall Ollama? [y/N]: "
    if /i "!DEL_OLLAMA!"=="y" (
        echo   [*] Uninstalling Ollama via winget ...
        winget uninstall --id Ollama.Ollama --silent 2>nul
        if errorlevel 1 (
            :: Try the built-in uninstaller
            if exist "%LocalAppData%\Programs\Ollama\unins000.exe" (
                echo   [*] Trying Ollama's built-in uninstaller ...
                "%LocalAppData%\Programs\Ollama\unins000.exe" /SILENT
                timeout /t 5 /nobreak >nul
            ) else (
                echo   [WARNING] Could not auto-uninstall. Remove Ollama from Settings ^> Apps.
            )
        )
        echo   [OK] Ollama uninstall initiated.
        set /a TOTAL_REMOVED+=1

        :: Clean up .ollama config folder
        if exist "%USERPROFILE%\.ollama" (
            echo.
            echo   [^^!] Ollama config folder remains: %USERPROFILE%\.ollama
            if !SILENT! equ 1 (set DEL_OLLAMA_DIR=) else set /p DEL_OLLAMA_DIR="      Delete entire .ollama folder (config + SSH keys)? [y/N]: "
            if /i "!DEL_OLLAMA_DIR!"=="y" (
                rmdir /s /q "%USERPROFILE%\.ollama" 2>nul
                echo   [OK] .ollama folder deleted.
            ) else (
                echo   [*] Skipped.
            )
        )
    ) else (
        echo   [*] Skipped.
    )
) else (
    echo   [OK] Ollama not found in standard locations.
)

:: ════════════════════════════════════════════════════════════════════════════
:: 7. Uninstall Python (if installed by setup.bat via winget)
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 7 of 8: Uninstall Python
echo ============================================================
echo.
echo   WARNING: Python may be used by other applications on your system.
echo   Only uninstall if setup.bat installed it and nothing else needs it.
echo.
if !SILENT! equ 1 (set DEL_PYTHON=) else set /p DEL_PYTHON="      Uninstall Python? [y/N]: "
if /i "!DEL_PYTHON!"=="y" (
    echo   [*] Attempting to uninstall Python via winget ...
    winget uninstall --id Python.Python.3.13 --silent 2>nul
    if errorlevel 1 (
        winget uninstall --id Python.Python.3.12 --silent 2>nul
        if errorlevel 1 (
            echo   [WARNING] Could not auto-uninstall Python. Remove from Settings ^> Apps.
        ) else (
            echo   [OK] Python 3.12 uninstall initiated.
            set /a TOTAL_REMOVED+=1
        )
    ) else (
        echo   [OK] Python 3.13 uninstall initiated.
        set /a TOTAL_REMOVED+=1
    )
) else (
    echo   [*] Skipped.
)

:: ════════════════════════════════════════════════════════════════════════════
:: 8. LocalAI application folder
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Step 8 of 8: LocalAI Application Folder
echo   Location: %~dp0
echo ============================================================
echo.
echo   This removes the LocalAI Studio application files, config,
echo   logs, benchmark results, and this uninstall script itself.
echo.
if !SILENT! equ 1 (set DEL_APP=) else set /p DEL_APP="      Delete the entire LocalAI folder? [y/N]: "
if /i "!DEL_APP!"=="y" (
    :: Clean up generated files first
    if exist "%~dp0python_path.bat" del /q "%~dp0python_path.bat" 2>nul
    if exist "%~dp0comfyui_path.bat" del /q "%~dp0comfyui_path.bat" 2>nul
    if exist "%~dp0localai.log" del /q "%~dp0localai.log" 2>nul
    if exist "%~dp0comfyui.log" del /q "%~dp0comfyui.log" 2>nul
    if exist "%~dp0__pycache__" rmdir /s /q "%~dp0__pycache__" 2>nul
    if exist "%~dp0src\__pycache__" rmdir /s /q "%~dp0src\__pycache__" 2>nul
    if exist "%~dp0benchmark_results" rmdir /s /q "%~dp0benchmark_results" 2>nul

    :: Strip trailing backslash from %~dp0 for rmdir
    set APP_DIR=%~dp0
    if "!APP_DIR:~-1!"=="\" set APP_DIR=!APP_DIR:~0,-1!

    :: Get parent directory (to remove it too if empty after app is gone)
    for %%I in ("!APP_DIR!") do set APP_PARENT=%%~dpI
    if "!APP_PARENT:~-1!"=="\" set APP_PARENT=!APP_PARENT:~0,-1!

    :: Write a tiny cleanup script to TEMP -- it runs after we exit and releases locks
    set CLEANUP_BAT=%TEMP%\localai_cleanup_%RANDOM%.bat
    echo @echo off> "!CLEANUP_BAT!"
    echo cd /d %%SystemRoot%%>> "!CLEANUP_BAT!"
    echo ping -n 4 127.0.0.1 ^>nul>> "!CLEANUP_BAT!"
    echo rmdir /s /q "!APP_DIR!" 2^>nul>> "!CLEANUP_BAT!"
    echo :: Remove parent folder if it is now empty>> "!CLEANUP_BAT!"
    echo dir /b "!APP_PARENT!" 2^>nul ^| findstr "." ^>nul 2^>^&1 ^|^| rmdir "!APP_PARENT!" 2^>nul>> "!CLEANUP_BAT!"
    echo del "%%~f0" 2^>nul>> "!CLEANUP_BAT!"
    start "" /b cmd /c ""!CLEANUP_BAT!""

    echo.
    echo   [OK] Generated files cleaned up.
    echo   [*] Folder will be removed automatically in a few seconds:
    echo       !APP_DIR!
    echo.
    set /a TOTAL_REMOVED+=1
) else (
    echo   [*] Skipped.
    echo.
    :: Still offer to clean generated files
    if !SILENT! equ 1 (set DEL_GEN=) else set /p DEL_GEN="      Delete just generated files (logs, caches, configs)? [y/N]: "
    if /i "!DEL_GEN!"=="y" (
        if exist "%~dp0python_path.bat" del /q "%~dp0python_path.bat" 2>nul
        if exist "%~dp0comfyui_path.bat" del /q "%~dp0comfyui_path.bat" 2>nul
        if exist "%~dp0localai.log" del /q "%~dp0localai.log" 2>nul
        if exist "%~dp0comfyui.log" del /q "%~dp0comfyui.log" 2>nul
        if exist "%~dp0__pycache__" rmdir /s /q "%~dp0__pycache__" 2>nul
        if exist "%~dp0src\__pycache__" rmdir /s /q "%~dp0src\__pycache__" 2>nul
        if exist "%~dp0benchmark_results" rmdir /s /q "%~dp0benchmark_results" 2>nul
        echo   [OK] Generated files removed.
        set /a TOTAL_REMOVED+=1
    ) else (
        echo   [*] Skipped.
    )
)

:: ════════════════════════════════════════════════════════════════════════════
:: Summary
:: ════════════════════════════════════════════════════════════════════════════
echo.
echo ============================================================
echo   Uninstall Complete
echo ============================================================
echo.
echo   Components processed: !TOTAL_REMOVED!
echo.
if !TOTAL_REMOVED! equ 0 (
    echo   Nothing was removed. Your system is unchanged.
) else (
    echo   Summary of locations that may still need manual cleanup:
    echo.
    echo     ComfyUI ^(standard^):  %LocalAppData%\LocalAI\ComfyUI
    echo     ComfyUI ^(legacy^):    ^<app folder^>\ComfyUI
    echo     ONNX models ^(new^):   %LocalAppData%\LocalAI\onnx  /  \ov
    echo     ONNX models ^(old^):   ^<app folder^>\models
    echo     Ollama models:       %USERPROFILE%\.ollama\models
    echo     Ollama config:       %USERPROFILE%\.ollama
    echo     HuggingFace cache:   %USERPROFILE%\.cache\huggingface
    echo     LocalAI app:         %~dp0
    echo.
    echo   Check each path above -- if the folder still exists and you
    echo   want it gone, delete it manually.
)
echo.
echo   Your system should now be restored to its pre-LocalAI state.
echo.
if !SILENT! equ 0 pause
goto :eof

:: ── Subroutine: detect and optionally stop Ollama ────────────────────────────
:handle_ollama
set OLLAMA_RUNNING=0

:: Check for ollama.exe process
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /i "ollama.exe" >nul 2>&1
if not errorlevel 1 set OLLAMA_RUNNING=1

:: Check for tray app process
tasklist /FI "IMAGENAME eq ollama app.exe" 2>nul | find /i "ollama app" >nul 2>&1
if not errorlevel 1 set OLLAMA_RUNNING=1

:: Check if Ollama Windows service is running (state 4 = RUNNING)
sc query ollama 2>nul | findstr /C:"RUNNING" >nul 2>&1
if not errorlevel 1 set OLLAMA_RUNNING=1

if !OLLAMA_RUNNING! equ 0 (
    echo   [OK] Ollama is not running.
    goto :eof
)

echo   [^!] Ollama is currently running.
if !SILENT! equ 1 (set STOP_OLLAMA=n) else set /p STOP_OLLAMA="      Stop Ollama now? [Y/n]: "
if /i "!STOP_OLLAMA!"=="n" (
    echo   [*] Skipped. Note: Ollama models cannot be removed while it is running.
    goto :eof
)

echo   [*] Stopping Ollama ...
net stop ollama /y >nul 2>&1
sc stop ollama >nul 2>&1
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM ollama.exe >nul 2>&1
ping -n 4 127.0.0.1 >nul
echo   [OK] Ollama stopped.
set /a PROCS_STOPPED+=1
goto :eof
