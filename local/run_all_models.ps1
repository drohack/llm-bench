# Run full test suite against every local model sequentially.
#
# WHY THIS SCRIPT EXISTS:
#   We are evaluating local quantized MoE models as potential drop-in replacements
#   for Claude Code via clawgate. The goal is to find a model that can pass Claude
#   Code's real tool-calling schema, run fast enough for interactive use (~40 tok/s
#   target), handle 128K+ context, and produce correct code.
#
# HARDWARE: RTX 3080 10 GB VRAM, 96 GB DDR4-3400 RAM (~54 GB/s bandwidth).
#   MoE expert layers are offloaded to RAM (--n-cpu-moe); only the always-active
#   backbone (attention + embeddings) lives on GPU. The KV cache is pre-allocated
#   in VRAM at startup using turboquant's turbo4/turbo3 format (~3.5-bit, 2-3x
#   smaller than the default q8_0).
#
# WHY THESE MODELS:
#
#   Qwen3.6-35B-A3B (released 2026-04-16, Apache 2.0)
#     Architecture: 35B total / 3B active per token (MoE, 256 experts)
#     Published scores: LiveCodeBench v6=80.4, SWE-bench=73.4, MCPMark=37.0,
#                       Terminal-Bench=51.5, MMLU-Pro=85.2, GPQA=86.0, AIME26=92.7
#     IQ3_XXS (12.8 GB): baseline quant, ~46 tok/s warm gen -- close to 40 tok/s target
#     IQ4_XS  (17.7 GB): better quant, ~32 tok/s warm gen -- slightly below target
#     Note: HumanEval/EvalPlus not reported by Alibaba -- our run generates new data
#
#   Qwen3.5-122B-A10B (released 2026-02-24, Apache 2.0)
#     Architecture: 122B total / 10B active per token (MoE)
#     Published scores: LiveCodeBench v6=78.9, SWE-bench=72.0, CodeForces=2100,
#                       Terminal-Bench=49.4, MMLU-Pro=86.7, GPQA=86.6
#     IQ2_XXS (36.6 GB): lowest viable quant, ~11 tok/s -- too slow for daily use
#     Purpose: test whether 10B active params improve quality ceiling vs 3B
#     Interesting: 35B scores *higher* than 122B on LiveCodeBench (80.4 vs 78.9)
#     because Qwen3.6 is specifically tuned for agentic/coding tasks
#
#   Removed models:
#     Llama 4 Scout: 17B active but only 3-4 tok/s gen -- not practical
#     Gemma 4 26B:   PASS on tool calling but 9.8 GB VRAM (risky), lower gen tok/s
#     8B models:     all failed tool calling (400 errors), removed
#
# WHY THESE TESTS:
#   1. Tool calling:  does the model correctly call Claude Code tools with the
#      real 86K-token prompt (242 tools, full system prompt + MCP servers)?
#   2. Speed bench:   cold TTFT (first session cost), warm gen tok/s (ongoing cost)
#   3. NIAH:          can the model recall info at 128K/256K context depth? Tests
#      the "lost in the middle" problem. Only 128K+ tested -- 86K system prompt
#      means 128K is the practical minimum for any real project context.
#   4. Coding (10 problems): basic pass/fail on easy-hard Python problems
#   5. EvalPlus (HumanEval+): standard 164-problem benchmark -- pass@1 rate is
#      directly comparable to published leaderboard scores
#
# WHY THESE SERVER FLAGS:
#   -ctk q8_0 -ctv q8_0     standard KV quant (turboquant turbo4/3 not in b9143)
#   -b 8192                  larger batch = faster prefill on 86K prompts
#   --reasoning off          disables <think> tokens (Qwen3 extended thinking)
#   --no-mmap --mlock        pin model in RAM, no swap
#   n-cpu-moe auto-tuned    turboquant's n=26 over-offloads in b9143; start at 10 and tune up
#
# Usage: .\run_all_models.ps1
Start-Transcript -Path (Join-Path $PSScriptRoot 'logs\run_all_models.log') -Append

$LLAMA_EXE    = "C:\llama_cpp\llama-server.exe"   # b9143 standard llama.cpp (has SWA fix; no turbo KV types)
$PYTHON       = (Join-Path $PSScriptRoot "..\\.venv\Scripts\python.exe")
$TEST_SCRIPT  = (Join-Path $PSScriptRoot "..\shared\test_real_tools.py")
$SPEED_SCRIPT = (Join-Path $PSScriptRoot "..\shared\bench_llm_speed.py")
$NIAH_SCRIPT    = (Join-Path $PSScriptRoot "..\shared\niah_test.py")
$CODE_SCRIPT    = (Join-Path $PSScriptRoot "..\shared\coding_test.py")
$EVALPLUS_SCRIPT = (Join-Path $PSScriptRoot "..\shared\evalplus_test.py")
$BATCH_SCRIPT   = (Join-Path $PSScriptRoot "..\shared\test_batch_compact.py")
$CTX_ACC_SCRIPT = (Join-Path $PSScriptRoot "..\shared\test_context_accuracy.py")
$FIXTURE      = (Join-Path $PSScriptRoot "..\shared\fixture_real_request.json")

# VRAM target band: tune n-cpu-moe until VRAM sits between these two values.
# Lower bound = use GPU efficiently; upper bound = keep headroom.
$VRAM_MAX_MB    = 9600   # 10240 - 640 MB headroom (KV cache on CPU frees room vs old turbo4 setup)
$VRAM_TARGET_MB = 8800   # aim to be at least this full (otherwise pull layers back to GPU)

$models = @(
    @{
        tag      = "qwen3.6-35b-iq3xxs-128k"
        path     = "C:\llama_cpp\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
        ngl      = 999; jinja = $false; ctx = 131072
        ncpumoe  = 24; ncpustep = 2; timeout = 900; runs = 3  # verified 2026-05-14
    },
    @{
        tag      = "qwen3.6-35b-iq3xxs-256k"
        path     = "C:\llama_cpp\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
        ngl      = 999; jinja = $false; ctx = 262144
        ncpumoe  = 31; ncpustep = 2; timeout = 1200; runs = 3  # verified 2026-05-14
    }
)

function Get-VramUsedMB {
    # Retry up to 3 times -- nvidia-smi can return blank while VRAM is mid-release
    for ($i = 0; $i -lt 3; $i++) {
        $raw = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        $val = ($raw | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1) -as [int]
        if ($val -gt 0) { return $val }
        Start-Sleep -Seconds 3
    }
    return 0
}

function Wait-Server {
    param([string]$PyExe, [int]$TimeoutSec = 600)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        & $PyExe -c "import httpx,sys; r=httpx.get('http://127.0.0.1:8081/health',timeout=3); sys.exit(0 if r.status_code==200 else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Stop-AllLlamaServers {
    # Kill every llama-server process and wait for VRAM to fully release
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 15
    # Verify nothing is still bound to 8081
    $still = Get-Process llama-server -ErrorAction SilentlyContinue
    if ($still) { $still | Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 5 }
}

function Start-LlamaServer {
    param($tag, $path, $ngl, $jinja, $ctx, $ncpumoe)
    # Only redirect stderr -- redirecting both stdout+stderr opens a new window in PS 5.1
    $errLog = Join-Path $PSScriptRoot "logs\$tag-server-err.log"
    $cmdArgs = "--model `"$path`" --n-gpu-layers $ngl --ctx-size $ctx --parallel 1 " +
               "--port 8081 --host 0.0.0.0 -ctk q8_0 -ctv q8_0 " +
               "--no-mmap --mlock -b 8192 --reasoning off"
                           # NOTE: benchmark uses --reasoning off (direct server access, no proxy).
                           # Production run_server.ps1 uses --reasoning auto so the rate proxy
                           # can inject /no_think (normal turns) or let model think (error turns).
    if ($ncpumoe -gt 0) { $cmdArgs += " --n-cpu-moe $ncpumoe" }
    if ($jinja)         { $cmdArgs += " --jinja" }
    return Start-Process -FilePath $LLAMA_EXE -ArgumentList $cmdArgs `
        -PassThru -NoNewWindow -RedirectStandardError $errLog
}

# -----------------------------------------------------------------------------

foreach ($m in $models) {
    $tag     = $m.tag
    $path    = $m.path
    $timeout = $m.timeout
    $runs    = $m.runs

    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host "  MODEL: $tag" -ForegroundColor Cyan
    Write-Host "=======================================" -ForegroundColor Cyan

    if (-not (Test-Path $path)) {
        Write-Warning "Model file not found: $path - skipping"
        continue
    }

    $ncpumoe  = $m.ncpumoe
    $proc     = $null
    $loaded   = $false
    $lastVram = -1

    for ($attempt = 1; $attempt -le 10; $attempt++) {
        Write-Host "Attempt $attempt - starting server (n-cpu-moe=$ncpumoe)..." -ForegroundColor Yellow
        Stop-AllLlamaServers
        $proc = Start-LlamaServer -tag $tag -path $path -ngl $m.ngl `
                    -jinja $m.jinja -ctx $m.ctx -ncpumoe $ncpumoe

        Write-Host "Waiting for model to load (up to 10 min)..."
        if (-not (Wait-Server -PyExe $PYTHON -TimeoutSec 600)) {
            Write-Warning "Server did not become ready - skipping model"
            Stop-AllLlamaServers
            break
        }
        # Wait for VRAM to fully settle -- health responds before warmup finishes copying layers to GPU
        Write-Host "Settling VRAM (20s)..." -NoNewline
        Start-Sleep -Seconds 20
        Write-Host " done"

        $vram = Get-VramUsedMB
        $color = if ($vram -le $VRAM_MAX_MB) { "Green" } else { "Red" }
        Write-Host "VRAM: $vram MB  (target: $VRAM_TARGET_MB-$VRAM_MAX_MB MB  n-cpu-moe=$ncpumoe)" -ForegroundColor $color

        # Accept if: in target band, no stepping configured, or VRAM plateaued (auto-fit taking over)
        $plateaued = ($lastVram -ge 0 -and [math]::Abs($vram - $lastVram) -lt 50)
        if (($vram -ge $VRAM_TARGET_MB -and $vram -le $VRAM_MAX_MB) -or $m.ncpustep -eq 0 -or $plateaued) {
            if ($plateaued) { Write-Host "VRAM plateaued at $vram MB (auto-fit active) - accepting" -ForegroundColor DarkCyan }
            $loaded = $true
            break
        }

        $lastVram = $vram
        Stop-AllLlamaServers

        if ($vram -gt $VRAM_MAX_MB) {
            $ncpumoe = [math]::Min(999, $ncpumoe + $m.ncpustep)
            Write-Host "Over budget - increasing n-cpu-moe to $ncpumoe" -ForegroundColor DarkYellow
        } else {
            $ncpumoe = [math]::Max(0, $ncpumoe - $m.ncpustep)
            Write-Host "Under target - decreasing n-cpu-moe to $ncpumoe" -ForegroundColor DarkCyan
        }
    }

    if (-not $loaded) {
        Write-Warning "Could not load $tag within VRAM budget - skipping"
        Stop-AllLlamaServers
        continue
    }

    Write-Host "Loaded OK  PID $($proc.Id)  VRAM $vram MB  n-cpu-moe=$ncpumoe" -ForegroundColor Green

    Write-Host ""
    Write-Host "--- Tool-calling test ---" -ForegroundColor Yellow
    & $PYTHON $TEST_SCRIPT --provider local --fixture $FIXTURE `
        --csv "$PSScriptRoot\results\tool_results.csv" `
        --models $tag --timeout $timeout

    Write-Host ""
    Write-Host "--- Speed benchmark: WITHOUT attribution header (CLAUDE_CODE_ATTRIBUTION_HEADER=0) ---" -ForegroundColor Yellow
    & $PYTHON $SPEED_SCRIPT `
        --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" `
        --runs $runs --max-tokens 800 `
        --fixture $FIXTURE --warmup `
        --gen-prompt "Without using any tools, write a detailed explanation of how you would implement a binary search tree in Python, including insert, search, and delete operations with full code examples." `
        --csv "$PSScriptRoot\results\speed_results.csv"

    Write-Host ""
    Write-Host "--- Speed benchmark: WITH attribution header (baseline / old behavior) ---" -ForegroundColor Yellow
    & $PYTHON $SPEED_SCRIPT `
        --model "$tag-with-header" --base-url "http://127.0.0.1:8081/v1" --api-key "none" `
        --runs $runs --max-tokens 800 `
        --fixture $FIXTURE --warmup --attribution-header `
        --gen-prompt "Without using any tools, write a detailed explanation of how you would implement a binary search tree in Python, including insert, search, and delete operations with full code examples." `
        --csv "$PSScriptRoot\results\speed_results.csv"

    Write-Host ""
    Write-Host "--- Needle in a Haystack ---" -ForegroundColor Yellow
    # 35B models (256K window): test 128K and 256K. 128K models: test 128K only.
    $niahCtx = if ($m.ctx -ge 200000) { @(128,256) } else { @(128) }
    & $PYTHON $NIAH_SCRIPT `
        --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" `
        --context-sizes $niahCtx `
        --depths 10 50 90 `
        --timeout $timeout `
        --csv "$PSScriptRoot\results\niah_results.csv"

    Write-Host ""
    Write-Host "--- Coding benchmark (10 problems) ---" -ForegroundColor Yellow
    & $PYTHON $CODE_SCRIPT `
        --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" `
        --timeout $timeout `
        --csv "$PSScriptRoot\results\coding_results.csv"

    Write-Host ""
    Write-Host "--- EvalPlus (HumanEval+) ---" -ForegroundColor Yellow
    & $PYTHON $EVALPLUS_SCRIPT `
        --model $tag --base-url "http://127.0.0.1:8081/v1" --api-key "none" `
        --csv "$PSScriptRoot\results\evalplus_results.csv"

    Write-Host ""
    Write-Host "--- Batch size compact benchmark (b=8192 baseline) ---" -ForegroundColor Yellow
    & $PYTHON $BATCH_SCRIPT --batch 8192 --rounds 2

    Write-Host ""
    Write-Host "--- Context accuracy + reasoning quality ---" -ForegroundColor Yellow
    $ctxK = if ($m.ctx -ge 200000) { 256 } else { 128 }
    & $PYTHON $CTX_ACC_SCRIPT --ctx $ctxK --depths 10 30 50 70 90

    Write-Host ""
    Write-Host "Stopping server..." -ForegroundColor Yellow
    Stop-AllLlamaServers
}

Write-Host ""
Write-Host "===== ALL DONE =====" -ForegroundColor Green
Write-Host ""
Write-Host "To run 256K context accuracy test (requires server restart):" -ForegroundColor DarkCyan
Write-Host "  1. Edit run_server.ps1: set --ctx-size 262144 and --n-cpu-moe 34" -ForegroundColor DarkCyan
Write-Host "  2. .\run_server.ps1" -ForegroundColor DarkCyan
Write-Host "  3. python ..\test_context_accuracy.py --ctx 256 --depths 10 30 50 70 90" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "To test b=16384 batch size for compact speed (requires server restart):" -ForegroundColor DarkCyan
Write-Host "  1. Edit run_server.ps1: set -b 16384 and --n-cpu-moe 26" -ForegroundColor DarkCyan
Write-Host "  2. .\run_server.ps1" -ForegroundColor DarkCyan
Write-Host "  3. python ..\test_batch_compact.py --batch 16384 --rounds 2" -ForegroundColor DarkCyan
Write-Host "Results: tool_results.csv  speed_results.csv  niah_results.csv  coding_results.csv  evalplus_results.csv" -ForegroundColor Green
Stop-Transcript
