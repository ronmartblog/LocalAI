@echo off
:: LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
setlocal enabledelayedexpansion
REM ─────────────────────────────────────────────────────────────────────────
REM  LocalAI Studio — Batch Benchmark Launcher
REM  Finds Python and Ollama, then runs run_batch.py with all CLI args.
REM ─────────────────────────────────────────────────────────────────────────

title LocalAI Studio - Batch Benchmark

REM ── Locate Python ──────────────────────────────────────────────────────
set "PYTHON="

REM Check for project-local venv first
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
    goto :found_python
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
    goto :found_python
)

REM Check python_path.bat (written by setup.bat)
if exist "%~dp0python_path.bat" (
    call "%~dp0python_path.bat"
    if defined LOCALAI_PYTHON (
        set "PYTHON=%LOCALAI_PYTHON%"
        goto :found_python
    )
    if defined PYTHON_PATH (
        set "PYTHON=%PYTHON_PATH%"
        goto :found_python
    )
)

REM Try standard names on PATH
where python >nul 2>&1 && (set "PYTHON=python" & goto :found_python)
where python3 >nul 2>&1 && (set "PYTHON=python3" & goto :found_python)
where py >nul 2>&1 && (set "PYTHON=py" & goto :found_python)

echo ERROR: Python not found. Please install Python 3.10+ or run setup.bat first.
pause
exit /b 1

:found_python
echo Using Python: %PYTHON%

REM ── Check Ollama ───────────────────────────────────────────────────────
where ollama >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama not found on PATH. Ollama tests will fail.
    echo          Install from https://ollama.com or add it to PATH.
    echo.
) else (
    REM Start Ollama serve if not already running
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
    if errorlevel 1 (
        echo Starting Ollama serve in background...
        start /min "" ollama serve
        timeout /t 3 /nobreak >nul
    ) else (
        echo Ollama is already running.
    )
)

REM ── Run batch benchmark ────────────────────────────────────────────────
echo.
echo ═══════════════════════════════════════════════════════
echo   LocalAI Studio — Batch Benchmark Mode
echo ═══════════════════════════════════════════════════════
echo.

REM v5.5.12: registry-reachability heads-up. Pure UX nicety — the actual
REM environment_skip decision is made inside run_batch.py / BatchRunner.
REM Surfaces an early warning instead of silently piling up ~3-minute pull
REM timeouts when the host has no Internet.
%PYTHON% -c "import socket,sys;socket.setdefaulttimeout(3);[socket.getaddrinfo('registry.ollama.ai',443,type=socket.SOCK_STREAM)];sys.exit(0)" >nul 2>&1
if errorlevel 1 (
    echo [doctor] registry.ollama.ai DNS lookup failed.
    echo          Models that are not already pulled locally will be skipped.
    echo          Pre-pull with 'ollama pull ^<tag^>' on a network-up machine,
    echo          or re-run once the connection is back.
    echo.
)

%PYTHON% "%~dp0run_batch.py" %*

echo.
if errorlevel 1 (
    echo Batch benchmark finished with errors.
) else (
    echo Batch benchmark complete.
)
pause
