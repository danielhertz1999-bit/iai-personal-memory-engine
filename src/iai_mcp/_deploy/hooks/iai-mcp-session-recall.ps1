# IAI-MCP SessionStart hook — recall injection (Windows).
#
# PowerShell equivalent of iai-mcp-session-recall.sh.
# Fires on Claude Code session start. Prints the cached session prefix
# to stdout for Claude Code to inject as additionalContext.
# Fail-safe: always exits 0 with empty stdout on any error.

$ErrorActionPreference = 'SilentlyContinue'

try {
    $inputText = [Console]::In.ReadToEnd()
} catch {
    $inputText = ''
}

$session_id = ''
$source_evt = ''
try {
    $obj = $inputText | ConvertFrom-Json
    $session_id = if ($obj.session_id) { $obj.session_id } else { '' }
    $source_evt = if ($obj.source) { $obj.source } else { '' }
} catch {}

$logDir = Join-Path $env:USERPROFILE '.iai-mcp\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logDate = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
$logFile = Join-Path $logDir "recall-$logDate.log"
$ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

Add-Content -Path $logFile -Value "---" -ErrorAction SilentlyContinue
Add-Content -Path $logFile -Value "$ts session=$session_id source=$source_evt" -ErrorAction SilentlyContinue

# Try the precache file first
$cachePath = Join-Path $env:USERPROFILE '.iai-mcp\.session-start-payload.cached.md'
if ((Test-Path $cachePath) -and (Get-Item $cachePath).Length -gt 0) {
    try {
        $cacheOut = Get-Content $cachePath -Raw -ErrorAction Stop
        if ($cacheOut.Length -gt 10000) { $cacheOut = $cacheOut.Substring(0, 10000) }
        if ($cacheOut) {
            [Console]::Out.Write($cacheOut)
            $cacheAge = [int]((Get-Date) - (Get-Item $cachePath).LastWriteTime).TotalSeconds
            Add-Content -Path $logFile -Value "$ts cache-hit age=${cacheAge}s bytes=$($cacheOut.Length)" -ErrorAction SilentlyContinue
            exit 0
        }
    } catch {}
    Add-Content -Path $logFile -Value "$ts cache-miss empty" -ErrorAction SilentlyContinue
} else {
    Add-Content -Path $logFile -Value "$ts cache-miss absent" -ErrorAction SilentlyContinue
}

# Find the iai-mcp CLI
$iai_cli = $null

if ($env:IAI_MCP_SESSION_RECALL_CLI -and (Test-Path $env:IAI_MCP_SESSION_RECALL_CLI)) {
    $iai_cli = $env:IAI_MCP_SESSION_RECALL_CLI
}

if (-not $iai_cli) {
    $cliCache = Join-Path $env:USERPROFILE '.iai-mcp\.cli-path'
    if (Test-Path $cliCache) {
        $cached = (Get-Content $cliCache -ErrorAction SilentlyContinue).Trim()
        if ($cached -and (Test-Path $cached)) { $iai_cli = $cached }
    }
}

if (-not $iai_cli) {
    try {
        $resolved = (Get-Command iai-mcp -ErrorAction Stop).Source
        if ($resolved) {
            $iai_cli = $resolved
            Set-Content -Path (Join-Path $env:USERPROFILE '.iai-mcp\.cli-path') -Value $iai_cli -ErrorAction SilentlyContinue
        }
    } catch {}
}

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

$usePythonModule = $false
if (-not $iai_cli) {
    try {
        $pyExe = (Get-Command python -ErrorAction Stop).Source
        $usePythonModule = $true
    } catch {}
}

if (-not $iai_cli -and -not $usePythonModule) {
    Add-Content -Path $logFile -Value "$ts skipped: iai-mcp CLI not found" -ErrorAction SilentlyContinue
    exit 0
}

# Run session-start with a 10s timeout
$hookTimeout = if ($env:IAI_MCP_RECALL_HOOK_TIMEOUT) { [int]$env:IAI_MCP_RECALL_HOOK_TIMEOUT } else { 10 }
$outTmp = Join-Path $logDir 'recall-stdout.tmp'

try {
    if ($usePythonModule) {
        $pyExe = (Get-Command python -ErrorAction Stop).Source
        $proc = Start-Process -FilePath $pyExe `
            -ArgumentList '-m', 'iai_mcp', 'session-start', '--session-id', $session_id `
            -NoNewWindow -PassThru -RedirectStandardOutput $outTmp -RedirectStandardError (Join-Path $logDir 'recall-stderr.tmp')
    } else {
        $proc = Start-Process -FilePath $iai_cli `
            -ArgumentList 'session-start', '--session-id', $session_id `
            -NoNewWindow -PassThru -RedirectStandardOutput $outTmp -RedirectStandardError (Join-Path $logDir 'recall-stderr.tmp')
    }
    $exited = $proc.WaitForExit($hookTimeout * 1000)
    if (-not $exited) {
        try { $proc.Kill() } catch {}
        $rc = 124
    } else {
        $rc = $proc.ExitCode
    }
} catch {
    $rc = 1
}

if ($rc -eq 0 -and (Test-Path $outTmp)) {
    $out = Get-Content $outTmp -Raw -ErrorAction SilentlyContinue
    if ($out) {
        [Console]::Out.Write($out)
    }
    $outLen = if ($out) { $out.Length } else { 0 }
} else {
    $outLen = 0
}

Remove-Item -Path $outTmp -Force -ErrorAction SilentlyContinue

Add-Content -Path $logFile -Value "$ts rc=$rc bytes=$outLen" -ErrorAction SilentlyContinue
exit 0
