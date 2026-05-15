# llm-bench

Benchmarks for local LLM and NVIDIA NIM inference with Claude Code.

## Structure

```
llm-bench/
├── shared/     — scripts used by both local and NVIDIA benches
├── local/      — local llama.cpp model benchmarks + results
└── nvidia/     — NVIDIA NIM cloud inference benchmarks + results
```

## Shared scripts

| Script | Purpose |
|--------|---------|
| `bench_llm_speed.py` | TTFT, prefill tok/s, gen tok/s — real Claude Code fixture |
| `niah_test.py` | Needle in a Haystack — recall at different context depths |
| `coding_test.py` | 10-problem coding benchmark (easy/medium/hard) |
| `evalplus_test.py` | HumanEval+ 164-problem benchmark (pass@1) |
| `test_real_tools.py` | Tool-calling with real 86K Claude Code fixture (242 tools) |
| `test_attribution_header.py` | KV cache impact of Claude Code billing header |
| `test_batch_compact.py` | Compact-scale cold prefill benchmark at different batch sizes |
| `test_context_accuracy.py` | NIAH + reasoning quality at 128K vs 256K context |
| `fixture_real_request.json` | Real 86K Claude Code request (242 tools, full system prompt) |

## Local results summary

Model: Qwen3.6-35B-A3B IQ3_XXS via llama.cpp b9143 on RTX 3080 10GB

| Metric | 128K | 256K |
|--------|------|------|
| Gen tok/s | 55.5 | 47.5 |
| Warm TTFT | 0.1s | 0.1s |
| Cold TTFT | 12.6s | 15.5s |
| Tool calling | PASS | — |
| NIAH | 100% | 100% |
| Coding | 90% (9/10) | — |
| EvalPlus | 92.7% (152/164) | — |

See `local/results/` for raw CSVs.

## NVIDIA NIM results summary

Models tested on NVIDIA NIM free tier via clawgate proxy.

See `nvidia/results/` for raw CSVs.

## Running benchmarks

### Local

```powershell
cd local
.\run_all_models.ps1   # full suite (auto-tunes n-cpu-moe, runs all tests)
.\run_bench.ps1        # quick bench against already-running server
```

### NVIDIA NIM

```powershell
cd nvidia
.\run_bench.ps1
.\run_clawgate_test.ps1
```
