# Main bench runner. Discovers live NVIDIA models, pings them, benches the top 6.
# Usage:
#   .\run_bench.ps1               # auto-discover top 6 >=100B
#   .\run_bench.ps1 -Top 10       # bench more
#   .\run_bench.ps1 -MinParams 70 # widen the net
#
# To bench specific models instead of auto-discovering:
#   python eval.py --models mistralai/mistral-medium-3.5-128b qwen/qwen3.5-122b-a10b

param(
    [int]$Top       = 6,
    [int]$MinParams = 100,
    [int]$Runs      = 3
)

$ErrorActionPreference = "Stop"
Start-Transcript -Path (Join-Path $PSScriptRoot 'logs\run_bench.log') -Append

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $python (Join-Path $PSScriptRoot "..\shared\eval.py") --top $Top --min-params $MinParams --runs $Runs

Stop-Transcript
