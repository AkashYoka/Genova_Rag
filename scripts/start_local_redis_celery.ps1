# Start native Redis + Celery for local Genova_Rag (no Docker).
# Compatible with Windows PowerShell 5.1 and PowerShell 7+.
$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { Split-Path $PSScriptRoot -Parent } else { "C:\PC\Yoka\Genova_Rag" }
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Backend ".venv\Scripts\python.exe"
$LogDir = Join-Path $Backend "data"
$CeleryLog = Join-Path $LogDir "celery_worker.log"
$CeleryErr = Join-Path $LogDir "celery_worker.err"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONPATH = $Backend
$env:CELERY_BROKER_URL = "redis://localhost:6379/0"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
$env:REDIS_URL = "redis://localhost:6379/2"
$env:CELERY_TASK_ALWAYS_EAGER = "false"

Write-Host "Checking Redis on :6379 ..."
$redisUp = $false
try {
    & $Python -c "import redis; raise SystemExit(0 if redis.from_url('redis://localhost:6379/0').ping() else 1)"
    if ($LASTEXITCODE -eq 0) { $redisUp = $true }
} catch {
    $redisUp = $false
}

if (-not $redisUp) {
    $cmd = Get-Command redis-server -ErrorAction SilentlyContinue
    if ($cmd) {
        $redisServer = $cmd.Source
    } else {
        $redisServer = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\redis-server.exe"
    }
    if (-not (Test-Path $redisServer)) {
        throw "redis-server not found. Install with: winget install taizod1024.redis-windows-fork"
    }
    Write-Host "Starting redis-server ..."
    Start-Process -FilePath $redisServer -ArgumentList "--port", "6379" -WindowStyle Minimized
    Start-Sleep -Seconds 2
}

& $Python -c "import redis; assert redis.from_url('redis://localhost:6379/0').ping(); print('Redis OK')"

$existing = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "celery_app\.celery_app" })
if ($existing.Count -gt 0) {
    $pids = ($existing | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Host "Celery already running (PID $pids)"
} else {
    Write-Host "Starting Celery worker ..."
    Start-Process -FilePath $Python `
        -ArgumentList "-m", "celery", "-A", "app.celery_app.celery_app", "worker", "--loglevel=info", "--pool=solo", "--hostname=celery@local" `
        -WorkingDirectory $Backend `
        -RedirectStandardOutput $CeleryLog `
        -RedirectStandardError $CeleryErr `
        -WindowStyle Hidden
    Start-Sleep -Seconds 8
}

$ping = & $Python -c "from app.celery_app import celery_app; print(celery_app.control.ping(timeout=5.0))"
Write-Host $ping
if ($ping -notmatch "pong") {
    Write-Host "Celery ping failed. Check logs:"
    Write-Host "  $CeleryLog"
    Write-Host "  $CeleryErr"
    if (Test-Path $CeleryErr) { Get-Content $CeleryErr -Tail 30 }
    exit 1
}
Write-Host "Done. Redis + Celery are up. Restart your local API if health still shows celery down."
