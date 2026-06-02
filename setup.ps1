# setup.ps1 — LocalAI Studio
#
# Thin PowerShell wrapper that:
#   1. Captures all setup output to setup.log via Start-Transcript so when
#      the console window closes (auto-close, accidental keypress, or
#      a missed pause on an early-exit path) the diagnostic record
#      survives.
#   2. Forces an unconditional "Press Enter to close" prompt at the end,
#      so even if the installer hits an exit path with no pause, the user
#      sees the outcome and the setup.log path before the window closes.
#
# The real installer lives in setup1.bat. setup.bat is a tiny shim that
# launches this script, which in turn launches setup1.bat under
# Start-Transcript. From the user's perspective they still double-click
# setup.bat and the experience is identical — they just get a setup.log
# captured automatically.
#
# This wrapper deliberately runs setup1.bat OUTSIDE the cmd process that
# started us. The v2026.06.01.4 self-tee that lived INSIDE setup.bat
# broke interactive `set /p` prompts because piping cmd's stdout through
# PowerShell switches Windows to block buffering. Start-Transcript
# captures from the host console without altering setup1.bat's stdout
# handles, so prompts work normally.
#
# Usage (typically invoked by setup.bat shim, but works standalone too):
#   .\setup.ps1
#   .\setup.ps1 all
#
# The setup.log file is written next to this script. Share it when asking
# for help diagnosing a failed install.

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$logPath   = Join-Path $scriptDir "setup.log"
$batPath   = Join-Path $scriptDir "setup1.bat"

if (-not (Test-Path -LiteralPath $batPath)) {
    Write-Host "[ERROR] setup1.bat not found next to setup.ps1." -ForegroundColor Red
    Write-Host "        Expected: $batPath" -ForegroundColor Red
    Write-Host "        setup1.bat holds the actual install logic; setup.bat is" -ForegroundColor Red
    Write-Host "        only a shim that launches this script. Restore setup1.bat" -ForegroundColor Red
    Write-Host "        from the release zip." -ForegroundColor Red
    Write-Host "Press Enter to close..." -ForegroundColor Yellow
    [void](Read-Host)
    exit 1
}

# Start-Transcript captures everything written to the host console, including
# the stdout/stderr of cmd /c children, so this single call records both the
# wrapper's banner text and setup.bat's entire output. -Force overwrites any
# previous setup.log so each run starts from a clean slate.
$transcriptStarted = $false
try {
    Start-Transcript -Path $logPath -Force | Out-Null
    $transcriptStarted = $true
} catch {
    Write-Host "[WARNING] Could not start transcript at $logPath." -ForegroundColor Yellow
    Write-Host "          Setup will still run, but no setup.log will be written." -ForegroundColor Yellow
    Write-Host "          Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  LocalAI Studio setup" -ForegroundColor Cyan
if ($transcriptStarted) {
    Write-Host "  Output is being captured to:" -ForegroundColor Cyan
    Write-Host "    $logPath" -ForegroundColor Cyan
    Write-Host "  Share this file if you need help diagnosing setup issues." -ForegroundColor Cyan
} else {
    Write-Host "  WARNING: transcript could not be started; no setup.log will be written." -ForegroundColor Yellow
}
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$exitCode = 0
try {
    # cmd /c launches setup1.bat in the same console so its `set /p` prompts,
    # winget interactive output, and pip progress bars all behave exactly as
    # they would for a direct double-click. The arguments after $batPath are
    # forwarded so callers can still pass "all" or other flags.
    $batArgs = if ($args -and $args.Count -gt 0) { " " + ($args -join " ") } else { "" }
    & cmd /c "`"$batPath`"$batArgs"
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) { $exitCode = 0 }
} catch {
    Write-Host ""
    Write-Host "[ERROR] setup1.bat invocation failed: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    Write-Host ""
    Write-Host "----------------------------------------------------------------" -ForegroundColor DarkGray
    if ($exitCode -eq 0) {
        Write-Host "  Setup exited with code 0 (no errors reported)." -ForegroundColor Green
    } else {
        Write-Host "  Setup exited with code $exitCode (errors reported above)." -ForegroundColor Red
    }
    if ($transcriptStarted) {
        Write-Host "  Full log saved to: $logPath" -ForegroundColor Green
    }
    Write-Host "----------------------------------------------------------------" -ForegroundColor DarkGray
    if ($transcriptStarted) {
        try { Stop-Transcript | Out-Null } catch {}
    }
    # Unconditional pause so the window stays open even if setup1.bat hit
    # an early exit path with no pause. Replaces the v.4 in-bat tee that
    # caused interactive prompt regressions.
    Write-Host ""
    Write-Host "Press Enter to close this window..." -ForegroundColor Yellow
    [void](Read-Host)
}

exit $exitCode
