# Run full test suite against whichever model is currently loaded in llama-server.
# Make sure run_server.ps1 is already running before executing this.
Start-Transcript -Path (Join-Path $PSScriptRoot 'logs\run_bench.log') -Append

$BASE_URL = "http://localhost:8081/v1"
$API_KEY  = "none"
$MODEL    = "local-model"   # llama-server ignores the model name
$RUNS     = 2
$VENV     = "..\\.venv\\Scripts\\python.exe"
$FIXTURE  = "..\shared\fixture_real_request.json"

Write-Host "=== Checking server ===" -ForegroundColor Cyan
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8081/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "Server up: $($resp.StatusCode)" -ForegroundColor Green
} catch {
    Write-Error "llama-server not responding on :8081 -- run .\run_server.ps1 first"
    exit 1
}

Write-Host ""
Write-Host "=== Tool-calling test (86K-token fixture, 242 tools) ===" -ForegroundColor Cyan
& $VENV ..\shared\test_real_tools.py --provider local `
    --fixture $FIXTURE `
    --csv (Join-Path $PSScriptRoot 'results\tool_results.csv')

Write-Host ""
Write-Host "=== Speed benchmark (fixture warmup + gen prompt) ===" -ForegroundColor Cyan
& $VENV ..\shared\bench_llm_speed.py `
    --model $MODEL --base-url $BASE_URL --api-key $API_KEY `
    --runs $RUNS --max-tokens 800 `
    --fixture $FIXTURE --warmup `
    --gen-prompt "Without using any tools, write a detailed explanation of how you would implement a binary search tree in Python, including insert, search, and delete operations with full code examples." `
    --csv (Join-Path $PSScriptRoot 'results\speed_results.csv')

Write-Host ""
Write-Host "=== Needle in a Haystack (context window test) ===" -ForegroundColor Cyan
& $VENV ..\shared\niah_test.py `
    --model $MODEL --base-url $BASE_URL --api-key $API_KEY `
    --context-sizes 8 32 64 --depths 10 50 90 `
    --csv (Join-Path $PSScriptRoot 'results\niah_results.csv')

Write-Host ""
Write-Host "=== Coding benchmark (10 problems) ===" -ForegroundColor Cyan
& $VENV ..\shared\coding_test.py `
    --model $MODEL --base-url $BASE_URL --api-key $API_KEY `
    --csv (Join-Path $PSScriptRoot 'results\coding_results.csv')

Write-Host ""
Write-Host "Done. Results in results\" -ForegroundColor Green
Stop-Transcript
