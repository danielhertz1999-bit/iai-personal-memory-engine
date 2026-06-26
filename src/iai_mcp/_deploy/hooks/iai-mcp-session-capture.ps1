# IAI-MCP Stop hook — ambient WRITE-side capture (Windows).
#
# PowerShell equivalent of iai-mcp-session-capture.sh.
# Fires when a Claude Code session ends. Calls `iai-mcp capture-transcript
# --no-spawn` to batch-capture the session transcript.
# Fail-safe: always exits 0.

$ErrorActionPreference = 'SilentlyContinue'

try {
    $inputText = [Console]::In.ReadToEnd()
} catch {
    $inputText = ''
}

$session_id = ''
$transcript_path = ''
$cwd = ''
try {
    $obj = $inputText | ConvertFrom-Json
    $session_id = if ($obj.session_id) { $obj.session_id } else { '' }
    $transcript_path = if ($obj.transcript_path) { $obj.transcript_path } else { '' }
    $cwd = if ($obj.cwd) { $obj.cwd } else { '' }
} catch {}

# Fallback: locate transcript if the hook payload didn't include its path.
if (-not $transcript_path -and $session_id) {
    $projectsDir = Join-Path $env:USERPROFILE '.claude\projects'
    if (Test-Path $projectsDir) {
        Get-ChildItem -Path $projectsDir -Directory | ForEach-Object {
            $candidate = Join-Path $_.FullName "$session_id.jsonl"
            if ((Test-Path $candidate) -and -not $transcript_path) {
                $transcript_path = $candidate
            }
        }
    }
}

$logDir = Join-Path $env:USERPROFILE '.iai-mcp\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logDate = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
$logFile = Join-Path $logDir "capture-$logDate.log"
$ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

Add-Content -Path $logFile -Value "---" -ErrorAction SilentlyContinue
Add-Content -Path $logFile -Value "$ts session=$session_id cwd=$cwd transcript=$transcript_path" -ErrorAction SilentlyContinue

if (-not $transcript_path -or -not (Test-Path $transcript_path)) {
    Add-Content -Path $logFile -Value "$ts skipped: no transcript found" -ErrorAction SilentlyContinue
    exit 0
}

# Rename the active-writer marker so the drain can see it.
if ($session_id) {
    $liveFile = Join-Path $env:USERPROFILE ".iai-mcp\.deferred-captures\$session_id.live.jsonl"
    if (Test-Path $liveFile) {
        $epoch = [int][double]::Parse((Get-Date -UFormat '%s'))
        $newName = "$session_id.live-$epoch.jsonl"
        $destDir = Split-Path $liveFile -Parent
        Move-Item -Path $liveFile -Destination (Join-Path $destDir $newName) -Force -ErrorAction SilentlyContinue
    }
    $offsetState = Join-Path $env:USERPROFILE ".iai-mcp\.capture-state\$session_id.offset"
    if (Test-Path $offsetState) { Remove-Item -Path $offsetState -Force -ErrorAction SilentlyContinue }
}

# Find the iai-mcp CLI
$iai_cli = $null

# 1. Environment variable override
if ($env:IAI_MCP_SESSION_CAPTURE_CLI -and (Test-Path $env:IAI_MCP_SESSION_CAPTURE_CLI)) {
    $iai_cli = $env:IAI_MCP_SESSION_CAPTURE_CLI
}

# 2. Cached CLI path
if (-not $iai_cli) {
    $cliCache = Join-Path $env:USERPROFILE '.iai-mcp\.cli-path'
    if (Test-Path $cliCache) {
        $cached = (Get-Content $cliCache -ErrorAction SilentlyContinue).Trim()
        if ($cached -and (Test-Path $cached)) { $iai_cli = $cached }
    }
}

# 3. PATH lookup
if (-not $iai_cli) {
    try {
        $resolved = (Get-Command iai-mcp -ErrorAction Stop).Source
        if ($resolved) {
            $iai_cli = $resolved
            Set-Content -Path (Join-Path $env:USERPROFILE '.iai-mcp\.cli-path') -Value $iai_cli -ErrorAction SilentlyContinue
        }
    } catch {}
}

# 4. Common Windows install locations
if (-not $iai_cli) {
    $candidates = @(
        (Join-Path $env:USERPROFILE '.local\bin\iai-mcp.exe'),
        (Join-Path $env:USERPROFILE 'IAI-MCP\.venv\Scripts\iai-mcp.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Scripts\iai-mcp.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $iai_cli = $c
            Set-Content -Path (Join-Path $env:USERPROFILE '.iai-mcp\.cli-path') -Value $iai_cli -ErrorAction SilentlyContinue
            break
        }
    }
}

# 5. Fall back to python -m iai_mcp
if (-not $iai_cli) {
    $pyExe = $null
    try { $pyExe = (Get-Command python -ErrorAction Stop).Source } catch {}
    if ($pyExe) {
        $iai_cli = "__python__"
    }
}

if (-not $iai_cli) {
    Add-Content -Path $logFile -Value "$ts skipped: iai-mcp CLI not found" -ErrorAction SilentlyContinue
    exit 0
}

# Run capture with a 30s timeout
try {
    if ($iai_cli -eq "__python__") {
        $pyExe = (Get-Command python -ErrorAction Stop).Source
        $proc = Start-Process -FilePath $pyExe `
            -ArgumentList '-m', 'iai_mcp', 'capture-transcript', '--no-spawn', '--session-id', $session_id, '--max-turns', '100000', $transcript_path `
            -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $logDir 'capture-stdout.tmp') -RedirectStandardError (Join-Path $logDir 'capture-stderr.tmp')
    } else {
        $proc = Start-Process -FilePath $iai_cli `
            -ArgumentList 'capture-transcript', '--no-spawn', '--session-id', $session_id, '--max-turns', '100000', $transcript_path `
            -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $logDir 'capture-stdout.tmp') -RedirectStandardError (Join-Path $logDir 'capture-stderr.tmp')
    }
    $exited = $proc.WaitForExit(30000)
    if (-not $exited) {
        try { $proc.Kill() } catch {}
    }
    $rc = if ($exited) { $proc.ExitCode } else { 124 }
} catch {
    $rc = 1
}

Add-Content -Path $logFile -Value "$ts rc=$rc" -ErrorAction SilentlyContinue
exit 0
