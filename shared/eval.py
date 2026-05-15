"""
eval.py -- Discover and benchmark NVIDIA models in one pass.

Flow:
  1. Fetch all >=100B models from NVIDIA (+ top-2 Qwen always included)
  2. Ping all in parallel -> live TTFT
  3. Take top --top N fastest (default 6)
  4. For each: run tool-calling test; skip speed bench if not PASS
  5. For PASS models: run speed benchmark
  6. Print summary table; results appended to results/ CSVs as normal

Usage:
  python eval.py                       # top 6 from >=100B models
  python eval.py --top 10              # bench more
  python eval.py --min-params 70       # widen the net
  python eval.py --runs 5              # more speed bench runs (default 3)
  python eval.py --models m1 m2 m3     # skip discovery, bench specific models
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx not installed. Run: pip install httpx openai tiktoken")

HERE      = Path(__file__).parent
FIXTURE   = HERE / "fixture_real_request.json"
TOOL_CSV  = HERE / "results" / "tool_results.csv"
SPEED_CSV = HERE / "results" / "speed_results.csv"

VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"
PYTHON  = str(VENV_PY) if VENV_PY.exists() else sys.executable

# Read API key + base URL from existing config in test_real_tools
def _load_nvidia_config() -> tuple[str, str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("trt", HERE / "test_real_tools.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.PROVIDERS["nvidia"]
    return cfg["api_key"], cfg["base_url"]


def _parse_params_b(model_id: str) -> int:
    hits = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)b", model_id.lower())]
    return int(max(hits)) if hits else 0


def _build_candidates(api_key: str, base_url: str, min_b: int) -> list[str]:
    r = httpx.get(f"{base_url}/models",
                  headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
    r.raise_for_status()
    skip = ("embed", "rerank", "vision", "reward", "tts", "whisper")
    all_ids = [m["id"] for m in r.json()["data"]
               if not any(kw in m["id"].lower() for kw in skip)]
    above = {m for m in all_ids if _parse_params_b(m) >= min_b}
    qwen  = sorted([m for m in all_ids if "qwen" in m.lower()],
                   key=_parse_params_b, reverse=True)
    return sorted(above | set(qwen[:2]), key=_parse_params_b, reverse=True)


def _ping(api_key: str, base_url: str, model: str) -> float | str:
    """Return TTFT float on success, or a short error string on failure."""
    try:
        t0 = time.monotonic()
        with httpx.stream(
            "POST", f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 5, "stream": True},
            timeout=20,
        ) as r:
            if r.status_code == 200:
                for _ in r.iter_bytes(1):
                    return time.monotonic() - t0
            return f"{r.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except Exception as e:
        return f"err: {type(e).__name__}"


def _ping_all(api_key: str, base_url: str, models: list[str]) -> dict[str, float | str]:
    with ThreadPoolExecutor(max_workers=min(len(models), 20)) as pool:
        futures = {pool.submit(_ping, api_key, base_url, m): m for m in models}
        return {futures[f]: f.result() for f in as_completed(futures)}


def _last_rows(path: Path, model: str, n: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("model") == model][-n:]


def _run_tool_test(model: str) -> str:
    subprocess.run(
        [PYTHON, str(HERE / "test_real_tools.py"),
         "--provider", "nvidia",
         "--models", model,
         "--fixture", str(FIXTURE),
         "--csv", str(TOOL_CSV),
         "--timeout", "120"],
    )
    rows = _last_rows(TOOL_CSV, model, 1)
    return rows[0]["verdict"] if rows else "ERROR"


def _run_speed_bench(model: str, runs: int) -> float | None:
    subprocess.run(
        [PYTHON, str(HERE / "bench_llm_speed.py"),
         "--model", model,
         "--runs", str(runs),
         "--csv", str(SPEED_CSV)],
    )
    rows = _last_rows(SPEED_CSV, model, runs)
    vals = [float(r["tokens_per_second_usage"]) for r in rows
            if r.get("tokens_per_second_usage")]
    return round(sum(vals) / len(vals), 1) if vals else None


def main():
    p = argparse.ArgumentParser(description="Discover and benchmark NVIDIA models")
    p.add_argument("--top",        type=int,    default=6,
                   help="Models to bench after ping (default: 6)")
    p.add_argument("--min-params", type=int,    default=100,
                   help="Min size in B for discovery (default: 100)")
    p.add_argument("--runs",       type=int,    default=3,
                   help="Speed bench runs per PASS model (default: 3)")
    p.add_argument("--models",     nargs="+",
                   help="Skip discovery; bench these specific models")
    args = p.parse_args()

    api_key, base_url = _load_nvidia_config()

    # ── 1. Discover / ping ────────────────────────────────────────────────────
    if args.models:
        candidates = args.models
        print(f"Pinging {len(candidates)} specified models...\n")
    else:
        print("Fetching NVIDIA catalog...")
        candidates = _build_candidates(api_key, base_url, args.min_params)
        print(f"Found {len(candidates)} candidates >={args.min_params}B (+ top-2 Qwen). Pinging...\n")

    live = _ping_all(api_key, base_url, candidates)

    responding  = sorted([(m, t) for m, t in live.items() if isinstance(t, float)],
                         key=lambda x: x[1])
    errors      = [(m, t) for m, t in live.items() if isinstance(t, str)]
    top_models  = [m for m, _ in responding[:args.top]]

    print(f"  {'Model':<48} {'TTFT':<12} Params")
    print(f"  {'-'*48} {'-'*12} {'-'*6}")
    for m, t in responding:
        marker = "  <-- will bench" if m in top_models else ""
        print(f"  {m:<48} {t:.1f}s        {_parse_params_b(m)}B{marker}")
    for m, err in sorted(errors, key=lambda x: x[1]):
        print(f"  {m:<48} {err}")

    if not top_models:
        sys.exit("\nNo models responded successfully. Try again later.")

    # ── 2. Bench top N ───────────────────────────────────────────────────────
    print(f"\nBenching top {len(top_models)} models — tool test first, speed only on PASS.\n")

    results = []
    for model in top_models:
        ttft = live[model]
        print(f"{'=' * 62}")
        print(f"  {model}  ({ttft:.1f}s TTFT)")
        print(f"{'=' * 62}")

        print("  [1/2] Tool-calling test...")
        verdict = _run_tool_test(model)
        print(f"        Verdict: {verdict}")

        tok_s = None
        if verdict == "PASS":
            print(f"  [2/2] Speed benchmark ({args.runs} runs)...")
            tok_s = _run_speed_bench(model, args.runs)
            if tok_s:
                print(f"        Avg tok/s: {tok_s}")
        else:
            print("        Skipping speed bench.")
        results.append({"model": model, "ttft": ttft, "verdict": verdict, "tok_s": tok_s})
        print()

    # ── 3. Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<48} {'TTFT':<8} {'Tool':<8} tok/s")
    print(f"  {'-'*48} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        ttft_s = f"{r['ttft']:.1f}s"
        tok_s  = f"{r['tok_s']:.0f}" if r["tok_s"] else "--"
        print(f"  {r['model']:<48} {ttft_s:<8} {r['verdict']:<8} {tok_s}")
    print()
    print(f"Results appended to:")
    print(f"  {TOOL_CSV}")
    print(f"  {SPEED_CSV}")


if __name__ == "__main__":
    main()
