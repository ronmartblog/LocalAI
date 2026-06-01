@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
::
:: Relocate the Ollama models directory off the user-profile container.
::
:: On constrained GPU cloud VMs the user profile lives in
:: a roaming profile container (VHDX) with a fixed size cap (typically 30 GB)
:: regardless of how much physical disk is free on the underlying host.
:: When the container fills with Ollama blobs the second model's pull
:: fails with "no space left on device" even though Get-PSDrive reports
:: hundreds of GB free.  The fix is to point Ollama at a directory that
:: lives outside the roaming profile container — anywhere on local disk
:: that isn't under C:\Users\<you> works.
::
:: This script defaults the target to an "Ollama" subfolder right next to
:: the LocalAI app itself ("app path" = the folder this script lives in,
:: derived live from the batch macro %~dp0).  That keeps everything
:: LocalAI manages in one place — wherever the user unpacked the app —
:: and inherits its drive letter from the install location instead of
:: hardcoding a drive like D: that may not exist on every cloud VM.
::
:: This script:
::   1. Asks the user for a target directory (default: <app path>\Ollama).
::   2. Creates it if missing.
::   3. Sets OLLAMA_MODELS at the USER scope (persists across sessions
::      and survives roaming profile rotations).
::   4. Optionally moves any existing Ollama blobs into the new dir.
::   5. Tells the user to restart the Ollama daemon (we don't restart
::      automatically — Ollama may be running as the system service or
::      as the user's tray app and we don't want to guess).
::
:: SAFE TO RE-RUN: a second invocation overwrites the env var with the
:: new directory and re-prompts about migration.
::
:: Note: OLLAMA_MODELS is read by the Ollama daemon, which is shared
:: infrastructure on the machine.  After it restarts, EVERY app that
:: talks to Ollama on this PC (Open-WebUI, Continue.dev, other LLM
:: front-ends) will also start reading models from the new directory.

setlocal enabledelayedexpansion

:: ── Off-profile cache redirection (v5.3.10) ───────────────────────────────────
:: Same env block other helper bats use. Harmless duplication: the user may
:: launch this script directly without going through setup.bat / run.bat, and
:: every subsequent script that runs in the same window inherits these values
:: (the setlocal above keeps them scoped to this script's lifetime).
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
echo   Relocate Ollama Models Directory
echo ============================================================
echo.
echo This script points Ollama at a directory outside the profile
echo container by setting the OLLAMA_MODELS environment variable
echo for your user account.
echo.
echo Use this on constrained GPU cloud VMs when Ollama
echo model pulls fail with "no space left on device" even though
echo the underlying disk has plenty of free space — the roaming
echo profile container that holds C:\Users\%USERNAME%\.ollama
echo is full.
echo.
echo Default target is an "Ollama" subfolder next to the LocalAI
echo app (the folder this script lives in), so model blobs sit
echo on whatever drive you installed LocalAI to.
echo.

set "DEFAULT_TARGET=%~dp0Ollama"
:: Strip the trailing backslash that %~dp0 always appends so the
:: prompt and echo lines read cleanly (e.g. "C:\LocalAI\Ollama"
:: instead of "C:\LocalAI\\Ollama").
if "!DEFAULT_TARGET:~-2!"=="\\" set "DEFAULT_TARGET=!DEFAULT_TARGET:~0,-1!"

set /p TARGET="Target directory [%DEFAULT_TARGET%]: "
if "!TARGET!"=="" set "TARGET=%DEFAULT_TARGET%"

echo.
echo [*] Target: !TARGET!
if not exist "!TARGET!" (
    echo [*] Creating directory ...
    mkdir "!TARGET!" 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not create !TARGET!.
        echo         Check that the parent folder exists and you have
        echo         write permission to it.  The target must live
        echo         OUTSIDE the roaming profile container — anywhere
        echo         under your LocalAI install folder is a safe pick.
        pause
        exit /b 1
    )
)

echo.
echo [*] Setting OLLAMA_MODELS=!TARGET! at user scope ...
setx OLLAMA_MODELS "!TARGET!" >nul
if errorlevel 1 (
    echo [ERROR] setx failed.  OLLAMA_MODELS was not set.
    pause
    exit /b 1
)

:: Also push to the current cmd session so any subsequent step in the
:: same window sees the new value.  (setx writes the registry but does
:: not affect the running process tree.)
set "OLLAMA_MODELS=!TARGET!"

echo.
echo [*] Done.  Current state:
echo     OLLAMA_MODELS = !TARGET!
echo.
echo     Note: OLLAMA_MODELS is shared infrastructure.  After Ollama
echo     restarts, every other app on this PC that talks to Ollama
echo     (Open-WebUI, Continue.dev, other LLM front-ends) will also
echo     start reading models from !TARGET!.
echo.

set "OLD_DIR=%USERPROFILE%\.ollama\models"
if exist "!OLD_DIR!\blobs" (
    echo [*] Existing Ollama models found at:
    echo       !OLD_DIR!
    echo.
    set /p MOVE="Move existing blobs to the new directory? [y/N]: "
    if /i "!MOVE!"=="y" (
        echo [*] Moving — this can take several minutes per GB ...
        robocopy "!OLD_DIR!" "!TARGET!" /E /MOVE /NFL /NDL /NJH /NJS /R:1 /W:1
        if errorlevel 8 (
            echo [WARN] robocopy reported errors — review the messages above.
            echo         Some files may need to be deleted manually from !OLD_DIR!.
        ) else (
            echo [*] Moved.
        )
    ) else (
        echo [*] Skipped — existing blobs remain at !OLD_DIR! and will be
        echo     ignored after the Ollama restart.  Delete them manually
        echo     to reclaim roaming profile space.
    )
    echo.
)

echo ============================================================
echo   Next steps
echo ============================================================
echo   1. Close the Ollama tray app (right-click the llama icon -^>
echo      Quit Ollama) OR stop the Ollama service.
echo   2. Restart Ollama (Start menu -^> Ollama, or "net start ollama"
echo      if running as a service).
echo   3. Re-run your LocalAI benchmark or model download.
echo.
echo   To verify the new directory is in use after restart:
echo     ollama show llama3.2:1b ^| findstr /i models
echo   should reference !TARGET!.
echo.
pause
exit /b 0
