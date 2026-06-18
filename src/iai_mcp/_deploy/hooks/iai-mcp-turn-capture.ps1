# IAI-MCP UserPromptSubmit hook — per-turn ambient capture (Windows).
#
# PowerShell equivalent of iai-mcp-turn-capture.sh.
# Reads stdin JSON, extracts session_id + transcript_path, runs inline
# Python for low-latency capture. Fail-safe: always exits 0.

$ErrorActionPreference = 'SilentlyContinue'

try {
    $inputText = [Console]::In.ReadToEnd()
} catch {
    $inputText = ''
}

$session_id = ''
$transcript_path = ''
try {
    $obj = $inputText | ConvertFrom-Json
    $session_id = if ($obj.session_id) { $obj.session_id } else { '' }
    $transcript_path = if ($obj.transcript_path) { $obj.transcript_path } else { '' }
} catch {}

$logDir = Join-Path $env:USERPROFILE '.iai-mcp\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logDate = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
$logFile = Join-Path $logDir "turn-capture-$logDate.log"
$ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

if (-not $session_id -or -not $transcript_path) {
    Add-Content -Path $logFile -Value "$ts skipped: missing session_id or transcript_path" -ErrorAction SilentlyContinue
    exit 0
}

# Find python
$pyExe = $null
try { $pyExe = (Get-Command python -ErrorAction Stop).Source } catch {}
if (-not $pyExe) {
    try { $pyExe = (Get-Command python3 -ErrorAction Stop).Source } catch {}
}
if (-not $pyExe) {
    # Check common venv location
    $venvPy = Join-Path $env:USERPROFILE '.iai-mcp\.venv\Scripts\python.exe'
    if (Test-Path $venvPy) { $pyExe = $venvPy }
}
if (-not $pyExe) {
    Add-Content -Path $logFile -Value "$ts skipped: python not found" -ErrorAction SilentlyContinue
    exit 0
}

# Run the Python CLI for turn capture with a 5s timeout
try {
    $proc = Start-Process -FilePath $pyExe `
        -ArgumentList '-m', 'iai_mcp', 'capture-turn-deferred', '--session-id', $session_id, '--transcript-path', $transcript_path `
        -NoNewWindow -PassThru -RedirectStandardError (Join-Path $logDir 'turn-capture-stderr.tmp')
    $exited = $proc.WaitForExit(5000)
    if (-not $exited) {
        try { $proc.Kill() } catch {}
    }
    $rc = if ($exited) { $proc.ExitCode } else { 124 }
} catch {
    $rc = 1
}

Add-Content -Path $logFile -Value "$ts session=$session_id rc=$rc" -ErrorAction SilentlyContinue
exit 0
