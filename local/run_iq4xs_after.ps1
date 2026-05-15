# Waits for the current bench run to finish, then runs IQ4_XS with full VRAM tuning.

$LLAMA_EXE    = "C:\llama_cpp_turbo\llama-server.exe"
$PYTHON       = (Join-Path $PSScriptRoot "..\\.venv\Scripts\python.exe")
$TEST_SCRIPT  = (Join-Path $PSScriptRoot "..\test_real_tools.py")
$SPEED_SCRIPT = (Join-Path $PSScriptRoot "..\bench_llm_speed.py")
$NIAH_SCRIPT  = (Join-Path $PSScriptRoot "..\niah_test.py")
$CODE_SCRIPT  = (Join-Path $PSScriptRoot "..\coding_test.py")
$EVALPLUS_SCRIPT = (Join-Path $PSScriptRoot "..\evalplus_test.py")
$FIXTURE      = (Join-Path $PSScriptRoot "..\fixture_real_request.json")
$logPath      = Join-Path $PSScriptRoot "logs\run_all_models.log"

$VRAM_MAX_MB    = 9440
$VRAM_TARGET_MB = 8800

Write-Host "Waiting for current bench run to finish..."
$deadline = (Get-Date).AddHours(12)
while ((Get-Date) -lt $deadline) {
    if (Select-String -Path $logPath -Pattern "ALL DONE" -Quiet -ErrorAction SilentlyContinue) {
        Write-Host "Current run finished. Starting IQ4_XS..."
        break
    }
    Start-Sleep -Seconds 30
}

$tag     = "qwen3.6-35b-iq4xs"
$path    = "C:\llama_cpp\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
$timeout = 900
$ncpumoe = 28    # starting point: known to land near target band
$step    = 2     # fine-grained: ~500 MB per step for IQ4_XS
$loaded  = $false
$lastVram = -1

for ($attempt = 1; $attempt -le 10; $attempt++) {
    Write-Host "Attempt $attempt - starting server (n-cpu-moe=$ncpumoe)..."
    $outLog = Join-Path $PSScriptRoot "logs\$tag-server-out.log"
    $errLog = Join-Path $PSScriptRoot "logs\$tag-server-err.log"
    $cmdArgs = "--model `"$path`" --n-gpu-layers 999 --ctx-size 262144 --parallel 1 --port 8081 --host 0.0.0.0 -ctk turbo4 -ctv turbo3 --no-mmap --mlock -b 8192 --reasoning-budget 0 --n-cpu-moe $ncpumoe"
    $proc = Start-Process -FilePath $LLAMA_EXE -ArgumentList $cmdArgs -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    Write-Host "Waiting for server ready..."
    $dl = (Get-Date).AddSeconds(600)
    $ready = $false
    while ((Get-Date) -lt $dl) {
        try { $null = Invoke-RestMethod "http://localhost:8081/health" -TimeoutSec 3 -ErrorAction Stop; $ready = $true; break }
        catch { Start-Sleep -Seconds 5; Write-Host "." -NoNewline }
    }
    if (-not $ready) { Write-Warning "Server did not become ready"; $proc | Stop-Process -Force -ErrorAction SilentlyContinue; break }

    Write-Host " Settling VRAM (20s)..." -NoNewline
    Start-Sleep -Seconds 20
    Write-Host " done"

    $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim() -as [int]
    $color = if ($vram -le $VRAM_MAX_MB) { "Green" } else { "Red" }
    Write-Host "VRAM: $vram MB  (target: $VRAM_TARGET_MB-$VRAM_MAX_MB  n-cpu-moe=$ncpumoe)" -ForegroundColor $color

    $plateaued = ($lastVram -ge 0 -and [math]::Abs($vram - $lastVram) -lt 50)
    if (($vram -ge $VRAM_TARGET_MB -and $vram -le $VRAM_MAX_MB) -or $plateaued) {
        if ($plateaued) { Write-Host "VRAM plateaued at $vram MB - accepting" -ForegroundColor DarkCyan }
        $loaded = $true; break
    }

    $lastVram = $vram
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5

    if ($vram -gt $VRAM_MAX_MB) {
        $ncpumoe = [math]::Min(999, $ncpumoe + $step)
        Write-Host "Over budget - increasing n-cpu-moe to $ncpumoe" -ForegroundColor DarkYellow
    } else {
        $ncpumoe = [math]::Max(0, $ncpumoe - $step)
        Write-Host "Under target - decreasing n-cpu-moe to $ncpumoe" -ForegroundColor DarkCyan
    }
}

if (-not $loaded) { Write-Error "Could not load IQ4_XS within VRAM budget"; exit 1 }
Write-Host "Loaded OK  PID $($proc.Id)  VRAM $vram MB  n-cpu-moe=$ncpumoe" -ForegroundColor Green

Write-Host "--- Tool-calling test ---"
& $PYTHON $TEST_SCRIPT --provider local --fixture $FIXTURE --models $tag --timeout $timeout --csv "$PSScriptRoot\results\tool_results.csv"

Write-Host "--- Speed benchmark ---"
& $PYTHON $SPEED_SCRIPT --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" --runs 2 --max-tokens 800 --fixture $FIXTURE --warmup --gen-prompt "Without using any tools, write a detailed explanation of how you would implement a binary search tree in Python, including insert, search, and delete operations with full code examples." --csv "$PSScriptRoot\results\speed_results.csv"

Write-Host "--- Needle in a Haystack ---"
& $PYTHON $NIAH_SCRIPT --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" --context-sizes 128 256 --depths 10 50 90 --timeout $timeout --csv "$PSScriptRoot\results\niah_results.csv"

Write-Host "--- Coding benchmark ---"
& $PYTHON $CODE_SCRIPT --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" --timeout $timeout --csv "$PSScriptRoot\results\coding_results.csv"

Write-Host "--- EvalPlus ---"
& $PYTHON $EVALPLUS_SCRIPT --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" --csv "$PSScriptRoot\results\evalplus_results.csv"

Write-Host "Stopping server..."
if (-not $proc.HasExited) { $proc | Stop-Process -Force -ErrorAction SilentlyContinue }
Write-Host "IQ4_XS complete."
