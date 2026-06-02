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

setlocal

set "_SCRIPT_DIR=%~dp0"
set "_PS1=%_SCRIPT_DIR%setup.ps1"

if not exist "%_PS1%" (
    echo [ERROR] setup.ps1 not found next to setup.bat.
    echo         Expected: %_PS1%
    echo.
    echo The installer needs setup.ps1 to capture diagnostics to setup.log.
    echo If you customized your install, restore setup.ps1 from the release zip.
    echo.
    pause
    endlocal
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%_PS1%" %*
set "_EXIT=%ERRORLEVEL%"

endlocal & exit /b %_EXIT%
