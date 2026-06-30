# scripts\install.ps1 — first-time setup on Windows.
#
# Usage (from repo root, in a PowerShell prompt):
#   .\scripts\install.ps1
#
# Does:
#   1. Creates .venv if missing
#   2. Installs iai-mcp editable into the venv (builds Rust native engine)
#   3. Builds the TypeScript MCP wrapper
#   4. Adds .venv\Scripts to the user PATH so the CLI is callable globally
#   5. Optionally installs the daemon via Windows Task Scheduler
#
# Flags:
#   -DryRun        Print what would be done without doing it
#   -PurgeState    (uninstall.ps1 only) also remove ~/.iai-mcp state files
#
# Idempotent: safe to re-run.
#
# Requirements:
#   Python 3.11 or 3.12  (https://www.python.org/downloads/)
#   Node.js 18+          (https://nodejs.org/)
#   Rust toolchain       (https://rustup.rs/)
#   Visual C++ build tools (cargo needs the MSVC linker)

[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Step  { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok    { param($msg) Write-Host "   [OK] $msg" -ForegroundColor Green }
function Warn  { param($msg) Write-Host "   [!]  $msg" -ForegroundColor Yellow }
function Die   { param($msg) Write-Host "`n[FAIL] $msg" -ForegroundColor Red; exit 1 }

function Invoke-Step {
    param([string]$Desc, [scriptblock]$Block)
    if ($DryRun) { Ok "DRY-RUN: $Desc (skipped)"; return }
    & $Block
}

# Yes/No prompt that defaults to Yes. In -DryRun mode (e.g. CI) it never
# blocks on Read-Host — it auto-answers Yes so the flow is traced end to end
# without any interactive input or system changes.
function Confirm-Yes {
    param([string]$Prompt)
    if ($DryRun) { Ok "DRY-RUN: auto-yes — $Prompt"; return $true }
    $ans = Read-Host $Prompt
    return ($ans -match "^[Yy]?$")
}

Push-Location $RepoRoot

# ---------------------------------------------------------------------------
# 0. Preflight checks
# ---------------------------------------------------------------------------
Step "preflight checks"

$pythonCmd = $null
foreach ($candidate in @("python3.12", "python3.11", "python3", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $pythonCmd = $candidate
        break
    }
}
if (-not $pythonCmd) { Die "Python 3.11 or 3.12 not found. Install from https://www.python.org/downloads/" }

$pyVer = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVer -notin @("3.11", "3.12")) { Die "Python $pyVer found but 3.11 or 3.12 required." }
Ok "Python $pyVer at $((Get-Command $pythonCmd).Source)"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Die "Rust toolchain not found. Install from https://rustup.rs/ then restart PowerShell."
}
Ok "Rust: $((cargo --version 2>&1))"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Warn "Node.js not found — mcp-wrapper build will be skipped. Install from https://nodejs.org/"
    $hasNode = $false
} else {
    Ok "Node.js: $((node --version))"
    $hasNode = $true
}

# ---------------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------------
Step "python venv"
if (-not (Test-Path ".venv")) {
    Invoke-Step "create .venv" { & $pythonCmd -m venv .venv }
    Ok ".venv created"
} else {
    Ok ".venv already exists"
}

$pip = ".venv\Scripts\pip.exe"
$python = ".venv\Scripts\python.exe"

# ---------------------------------------------------------------------------
# 2. Editable install (builds Rust native engine via setuptools-rust)
# ---------------------------------------------------------------------------
Step "editable install (pip install -e .)"
Warn "The Rust native engine builds from source — this takes 2-5 minutes on first run."
Invoke-Step "pip install -e ." {
    & $pip install --quiet --upgrade pip
    & $pip install --quiet -e .
    if ($LASTEXITCODE -ne 0) { Die "pip install failed. See output above." }
}
Ok "iai-mcp Python package installed into .venv"

# ---------------------------------------------------------------------------
# 3. TypeScript MCP wrapper
# ---------------------------------------------------------------------------
Step "TS wrapper build"
if ($hasNode -and (Test-Path "mcp-wrapper")) {
    Invoke-Step "npm install + build" {
        Push-Location mcp-wrapper
        if (Test-Path "package-lock.json") {
            npm ci --silent --no-audit --no-fund
        } else {
            npm install --silent --no-audit --no-fund
        }
        npm run build --silent
        if ($LASTEXITCODE -ne 0) { Pop-Location; Die "npm build failed." }
        Pop-Location
    }
    Ok "mcp-wrapper\dist built"
} elseif (-not (Test-Path "mcp-wrapper")) {
    Warn "mcp-wrapper\ missing — skipping"
} else {
    Warn "Node.js not available — skipping mcp-wrapper build"
}

# ---------------------------------------------------------------------------
# 4. Add .venv\Scripts to user PATH (idempotent)
# ---------------------------------------------------------------------------
Step "user PATH update"
$venvScripts = Join-Path $RepoRoot ".venv\Scripts"
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User") -split ";"
if ($venvScripts -notin $userPath) {
    Invoke-Step "add .venv\Scripts to PATH" {
        $newPath = ($userPath + $venvScripts) -join ";"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    }
    Ok "Added $venvScripts to user PATH (restart your terminal to pick it up)"
} else {
    Ok "$venvScripts already in PATH"
}

# ---------------------------------------------------------------------------
# 5. Daemon — Windows Task Scheduler
# ---------------------------------------------------------------------------
Step "daemon installer (Windows Task Scheduler)"
$pythonwExe = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonwExe)) {
    $pythonwExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}

$taskName = "iai-mcp-daemon"
$existing = schtasks /Query /TN $taskName 2>&1
if ($LASTEXITCODE -eq 0) {
    Ok "Task Scheduler task '$taskName' already registered — skipping"
} else {
    Write-Host ""
    if (Confirm-Yes "Install iai-mcp daemon via Windows Task Scheduler? [Y/n]") {
        Invoke-Step "schtasks /Create" {
            $logDir = Join-Path $env:APPDATA "iai-mcp\logs"
            New-Item -ItemType Directory -Force -Path $logDir | Out-Null

            # Build the task XML so we can set Hidden=true and pass env vars cleanly.
            $taskXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>iai-mcp personal memory daemon</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <Hidden>true</Hidden>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$pythonwExe</Command>
      <Arguments>-m iai_mcp.daemon</Arguments>
      <WorkingDirectory>$RepoRoot</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
            $xmlPath = [System.IO.Path]::GetTempFileName() + ".xml"
            [System.IO.File]::WriteAllText($xmlPath, $taskXml, [System.Text.Encoding]::Unicode)
            schtasks /Create /TN $taskName /XML $xmlPath /F | Out-Null
            Remove-Item $xmlPath -ErrorAction SilentlyContinue
            if ($LASTEXITCODE -ne 0) { Die "schtasks /Create failed." }
        }
        Ok "Task Scheduler task '$taskName' registered (starts at logon)"

        Write-Host ""
        if (Confirm-Yes "Start the daemon now? [Y/n]") {
            Invoke-Step "schtasks /Run" { schtasks /Run /TN $taskName | Out-Null }
            Ok "Daemon started (Task: $taskName)"
        }
    } else {
        Warn "Skipped Task Scheduler install. Run manually: iai-mcp daemon install"
    }
}

# ---------------------------------------------------------------------------
# 6. Capture hooks
# ---------------------------------------------------------------------------
Step "capture + recall hooks"
Write-Host ""
if (Confirm-Yes "Install ambient capture+recall hooks into ~/.claude/settings.json? [Y/n]") {
    Invoke-Step "capture-hooks install" {
        & $python -m iai_mcp.cli capture-hooks install --yes
    }
    Ok "Hooks installed"
} else {
    Warn "Skipped. Run manually: iai-mcp capture-hooks install"
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Step "done"
Ok "iai-mcp installed successfully."
Write-Host ""
Write-Host "  Restart your terminal, then run:" -ForegroundColor White
Write-Host "    iai-mcp doctor      # full diagnostic" -ForegroundColor Gray
Write-Host "    iai capture 'hello' # test memory write" -ForegroundColor Gray
Write-Host "    iai recall 'hello'  # test memory recall" -ForegroundColor Gray
Write-Host ""

Pop-Location
