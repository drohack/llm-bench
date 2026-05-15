# Test each model's Claude Code tool calling ability through clawgate.
# Starts clawgate, runs the test, stops clawgate, moves to next model.
# Usage: .\run_clawgate_test.ps1

$ErrorActionPreference = "Stop"
$python  = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$script  = Join-Path $PSScriptRoot "..\shared\test_clawgate_tools.py"
$logOut  = Join-Path $PSScriptRoot "logs\clawgate.log"
$logErr  = Join-Path $PSScriptRoot "logs\clawgate.log.err"
$apiKey  = "your-nvidia-api-key"
$baseUrl = "https://integrate.api.nvidia.com/v1"
$port    = 8082

$models = @(
    "mistralai/mistral-nemotron"
    "qwen/qwen3-coder-480b-a35b-instruct"
    "mistralai/devstral-2-123b-instruct-2512"
    "z-ai/glm4.7"
    "stepfun-ai/step-3.5-flash"
)

function Wait-Port {
    param([int]$Port, [int]$TimeoutSec = 15)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", $Port)
            $tcp.Close()
            return $true
        } catch { Start-Sleep -Milliseconds 300 }
    }
    return $false
}

foreach ($model in $models) {
    Write-Host ""
    Write-Host "========== $model ==========" -ForegroundColor Cyan

    # Start clawgate with this model (all three tiers set to same model)
    $proc = Start-Process "clawgate" -ArgumentList @(
        "--mode=api",
        "--apiKey=$apiKey",
        "--baseUrl=$baseUrl",
        "--bigModel=$model",
        "--midModel=$model",
        "--smallModel=$model",
        "--port=$port"
    ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr

    Write-Host "  Started clawgate PID $($proc.Id), waiting for port $port..." -ForegroundColor DarkGray

    if (-not (Wait-Port -Port $port -TimeoutSec 15)) {
        Write-Host "  WARN: clawgate did not come up in time, skipping." -ForegroundColor Yellow
        $proc | Stop-Process -Force -ErrorAction SilentlyContinue
        continue
    }

    # Run the test
    & $python $script --model $model --port $port

    # Stop clawgate before next model
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500   # brief pause so port is released
}

Write-Host ""
Write-Host "All done. Results in bench\results\tool_calling_results.csv" -ForegroundColor Green
