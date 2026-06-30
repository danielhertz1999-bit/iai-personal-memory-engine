# scripts\uninstall.ps1 — Task Scheduler + daemon teardown on Windows.
#
# Usage:
#   .\scripts\uninstall.ps1               # remove Task Scheduler task + kill daemon
#   .\scripts\uninstall.ps1 -PurgeState   # also remove ~/.iai-mcp state files
#   .\scripts\uninstall.ps1 -PurgeData    # also remove ~/.iai-mcp/hippo (your brain) — DESTRUCTIVE
#
# Idempotent: safe to re-run. Never aborts mid-flow; each step reports its outcome.
# Inverse of scripts\install.ps1.

[CmdletBinding()]
param(
    [switch]$PurgeState,
    [switch]$PurgeData,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
# Deliberately no $ErrorActionPreference = "Stop" — best-effort cleanup must not abort.

$RepoRoot = Split-Path -Parent $PSScriptRoot
$IaiDir   = Join-Path $env:USERPROFILE ".iai-mcp"
$TaskName = "iai-mcp-daemon"

function Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok   { param($msg) Write-Host "   [OK] $msg" -ForegroundColor Green }
function Warn { param($msg) Write-Host "   [!]  $msg" -ForegroundColor Yellow }

if ($PurgeData) {
    Warn "-PurgeData is DESTRUCTIVE: $IaiDir\hippo (your brain) will be deleted!"
}

# ---------------------------------------------------------------------------
# 1. Stop Task Scheduler task
# ---------------------------------------------------------------------------
Step "stop Task Scheduler task"
$running = schtasks /Query /TN $TaskName /FO LIST 2>&1 | Select-String "Running"
if ($running) {
    if ($DryRun) {
        Ok "DRY-RUN: would stop task '$TaskName'"
    } else {
        schtasks /End /TN $TaskName 2>&1 | Out-Null
        Ok "Task '$TaskName' stopped"
    }
} else {
    Ok "Task '$TaskName' not running"
}

# ---------------------------------------------------------------------------
# 2. Remove Task Scheduler task
# ---------------------------------------------------------------------------
Step "remove Task Scheduler task"
$exists = schtasks /Query /TN $TaskName 2>&1
if ($LASTEXITCODE -eq 0) {
    if ($DryRun) {
        Ok "DRY-RUN: would delete task '$TaskName'"
    } else {
        schtasks /Delete /TN $TaskName /F 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { Ok "Task '$TaskName' removed" }
        else { Warn "schtasks /Delete returned $LASTEXITCODE — task may already be gone" }
    }
} else {
    Ok "Task '$TaskName' not registered (already clean)"
}

# ---------------------------------------------------------------------------
# 3. Kill any lingering daemon processes
# ---------------------------------------------------------------------------
Step "kill lingering daemon"
# Match the daemon by command line via WMI (the daemon runs hidden under
# pythonw.exe with no window, so Get-Process title matching won't find it).
# NB: $pid is a reserved read-only automatic variable in PowerShell, so the
# loop variable below is $procId, not $pid.
try {
    $wmiProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*iai_mcp.daemon*" }
    $procIds = @($wmiProcs | Select-Object -ExpandProperty ProcessId)
} catch {
    $procIds = @()
}

if ($procIds.Count -gt 0) {
    Warn "Found PIDs: $($procIds -join ', ')"
    foreach ($procId in $procIds) {
        if ($DryRun) { Ok "DRY-RUN: would Stop-Process -Id $procId" }
        else {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not $DryRun) { Ok "Lingering daemon process(es) terminated" }
} else {
    Ok "No lingering iai_mcp.daemon processes"
}

# ---------------------------------------------------------------------------
# 4. Remove IPC files (port file, token file, socket lock)
# ---------------------------------------------------------------------------
Step "remove IPC files"
$ipcFiles = @(
    (Join-Path $IaiDir ".daemon.port"),
    (Join-Path $IaiDir ".daemon.token"),
    (Join-Path $IaiDir ".daemon.sock"),
    (Join-Path $IaiDir ".lock")
)
foreach ($f in $ipcFiles) {
    if (Test-Path $f) {
        if ($DryRun) { Ok "DRY-RUN: would remove $f" }
        else { Remove-Item $f -Force -ErrorAction SilentlyContinue; Ok "Removed $f" }
    }
}
Ok "IPC files clean"

# ---------------------------------------------------------------------------
# 5. -PurgeState: remove state + pending-embeddings
# ---------------------------------------------------------------------------
if ($PurgeState) {
    Step "-PurgeState: remove daemon state files"
    $stateFiles = @(
        (Join-Path $IaiDir ".daemon-state.json"),
        (Join-Path $IaiDir "pending-embeddings")
    )
    foreach ($f in $stateFiles) {
        if (Test-Path $f) {
            if ($DryRun) { Ok "DRY-RUN: would remove $f" }
            else { Remove-Item $f -Recurse -Force -ErrorAction SilentlyContinue; Ok "Removed $f" }
        }
    }
    Ok "State files clean"
}

# ---------------------------------------------------------------------------
# 6. -PurgeData: remove hippo brain (DESTRUCTIVE)
# ---------------------------------------------------------------------------
if ($PurgeData) {
    Step "-PurgeData: remove $IaiDir\hippo (DESTRUCTIVE)"
    $ans = Read-Host "Really delete your memory store at $IaiDir\hippo? [y/N]"
    if ($ans -match "^[Yy]$") {
        if ($DryRun) { Ok "DRY-RUN: would remove $IaiDir\hippo" }
        else {
            $hippoDir = Join-Path $IaiDir "hippo"
            if (Test-Path $hippoDir) {
                Remove-Item $hippoDir -Recurse -Force -ErrorAction SilentlyContinue
                Ok "Memory store removed"
            } else {
                Ok "No memory store found at $hippoDir"
            }
        }
    } else {
        Warn "User declined — memory store preserved"
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Step "done"
Ok "iai-mcp uninstalled. Re-run .\scripts\install.ps1 to restore."
