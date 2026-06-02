@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
::
:: v2026.06.01.11: setup.bat is now a tiny shim that launches setup.ps1, which
:: starts a PowerShell transcript writing to setup.log and then invokes the real
:: installer (setup1.bat) inside that transcript. Users still double-click the
:: same setup.bat they always have — they just get an auto-captured setup.log
:: next to the app whether the window closes cleanly or auto-exits.
::
:: The PowerShell wrapper sits OUTSIDE this cmd process (it launches a fresh
:: powershell.exe and runs setup1.bat under it), so interactive `set /p`
:: prompts inside setup1.bat keep working — this is NOT the v.4-era self-tee
:: regression that block-buffered prompt text. See
:: tests\test_setup_release_contracts.py
:: ::test_setup_bat_does_not_use_powershell_tee_wrapper for the pin.
::
:: All args passed to setup.bat are forwarded through to setup1.bat unchanged.

setlocal enabledelayedexpansion

set "_SCRIPT_DIR=%~dp0"
set "_PS1=!_SCRIPT_DIR!setup.ps1"

rem === Zip-preview / temp-folder guard ============================================
rem v2026.06.02.0: Detect users who double-clicked setup.bat from inside Windows
rem Explorer's zip preview (path like %TEMP%\<GUID>_<zip>.zip[.e2b]\) or any
rem %TEMP% subfolder (7-Zip / WinRAR temp extractor). pip + the deeply-nested
rem CUDA wheels (onnxruntime_genai_cuda, torch, xformers) blow past Windows'
rem MAX_PATH=260 limit when the cache root is already buried in a long temp
rem path, producing a confusing "[Errno 2] No such file or directory" failure
rem mid-install. Fail fast with a friendly fix-it message instead.
rem
rem Uses cmd's substring substitution rather than `echo | findstr` because the
rem latter mis-parses a trailing backslash inside /C:"...\" and because the
rem set side of `findstr ... && set` runs in the pipe subshell (parent never
rem sees the assignment).
rem ==============================================================================
set "_FROM_ZIP_PREVIEW=0"
set "_FROM_TEMP_FOLDER=0"
if not "!_SCRIPT_DIR!"=="!_SCRIPT_DIR:.zip.e2b\=!" set "_FROM_ZIP_PREVIEW=1"
if not "!_SCRIPT_DIR!"=="!_SCRIPT_DIR:.zip\=!"     set "_FROM_ZIP_PREVIEW=1"
if defined TEMP (
    set "_TMP=!TEMP!\"
    call set "_STRIPPED=%%_SCRIPT_DIR:!_TMP!=%%"
    if not "!_SCRIPT_DIR!"=="!_STRIPPED!" set "_FROM_TEMP_FOLDER=1"
)
if defined LOCALAPPDATA (
    set "_TMP=!LOCALAPPDATA!\Temp\"
    call set "_STRIPPED=%%_SCRIPT_DIR:!_TMP!=%%"
    if not "!_SCRIPT_DIR!"=="!_STRIPPED!" set "_FROM_TEMP_FOLDER=1"
)
if not "!_FROM_ZIP_PREVIEW!!_FROM_TEMP_FOLDER!"=="00" (
    echo.
    echo ================================================================
    echo  [ERROR] LocalAI cannot install from inside a zip preview / temp folder.
    echo ================================================================
    echo.
    echo  setup.bat is running from:
    echo    !_SCRIPT_DIR!
    echo.
    echo  This looks like a Windows zip preview or temp-extractor location.
    echo  Installing from here WILL FAIL when pip unpacks the CUDA wheels
    echo  ^(Windows MAX_PATH=260 limit is exceeded by the deeply-nested
    echo  wheels under the cache directory^).
    echo.
    echo  To fix:
    echo    1. Close this window.
    echo    2. Right-click the LocalAI .zip in File Explorer and pick
    echo       "Extract All..." ^(do NOT just open / double-click the zip^).
    echo    3. Extract to a SHORT path like  C:\LocalAI
    echo    4. Open the extracted folder and double-click setup.bat there.
    echo.
    pause
    endlocal
    exit /b 1
)
rem === end zip-preview guard ====================================================

if not exist "!_PS1!" (
    echo [ERROR] setup.ps1 not found next to setup.bat.
    echo         Expected: !_PS1!
    echo.
    echo The installer needs setup.ps1 to capture diagnostics to setup.log.
    echo If you customized your install, restore setup.ps1 from the release zip.
    echo.
    pause
    endlocal
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "!_PS1!" %*
set "_EXIT=!ERRORLEVEL!"

endlocal & exit /b %_EXIT%
