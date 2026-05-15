"""
Three independent performance tests for local llama-server setup:

  Test 1: Attribution header impact at REALISTIC context size (~50K tokens)
  Test 2: Flash attention (-fa flag) throughput improvement
  Test 3: DISABLE_PROMPT_CACHING — does stripping cache_control blocks help?

Usage:
    python bench/test_perf_options.py [--test 1|2|3|all]
"""

import argparse, json, time, httpx, statistics, textwrap

LLAMA_URL = "http://127.0.0.1:8081"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def health_check():
    r = httpx.get(f"{LLAMA_URL}/health", timeout=5)
    r.raise_for_status()

def chat(messages: list[dict], max_tokens: int = 5) -> dict:
    t0 = time.monotonic()
    r = httpx.post(f"{LLAMA_URL}/v1/chat/completions",
                   json={"model": "local", "messages": messages, "max_tokens": max_tokens},
                   timeout=300)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    return {"elapsed_s": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)}

def prefill_rate(res):
    return res["prompt_tokens"] / res["elapsed_s"] if res["elapsed_s"] else 0

def gen_rate(res):
    pt, ct, e = res["prompt_tokens"], res["completion_tokens"], res["elapsed_s"]
    # Approximate: total time = prefill_time + gen_time
    # prefill_time ~ pt / 400 tok/s (rough baseline)
    gen_time = max(0.01, e - pt / 400)
    return ct / gen_time if gen_time else 0

def hdr(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")

def row(label, results):
    rates = [prefill_rate(r) for r in results]
    avg = statistics.mean(rates)
    vals = "  ".join(f"{v:.0f}" for v in rates)
    print(f"  {label:<40} {vals}   avg={avg:.0f} tok/s")
    return avg

# ---------------------------------------------------------------------------
# Shared prompt building
# ---------------------------------------------------------------------------

# ~500-token system body (one repetition)
BLOCK_500 = ("You are Claude Code, Anthropic's official CLI for Claude, "
             "running within the Claude Agent SDK. ") * 60

# ~50K token system body: repeat the block ~100x
BLOCK_50K = BLOCK_500 * 100

HISTORY = [
    {"role": "user",      "content": "Read src/index.ts and summarize it."},
    {"role": "assistant", "content": "The file exports an Express app with three routes: /health, /api/anilist, and /api/seasons."},
    {"role": "user",      "content": "What does the rate limiter do?"},
    {"role": "assistant", "content": "It caps requests at 100 per 15 minutes per IP and returns 429 JSON when exceeded."},
    {"role": "user",      "content": "Show me the seasons route handler."},
]

BILLING_HEADER = "x-anthropic-billing-header: cc_version=2.1.136; cc_entrypoint=claude-vscode; cch={cch};"

def msgs_with_header(system_body, cch):
    return [
        {"role": "system", "content": BILLING_HEADER.format(cch=cch)},
        {"role": "system", "content": system_body},
        *HISTORY,
    ]

def msgs_without_header(system_body):
    return [
        {"role": "system", "content": system_body},
        *HISTORY,
    ]

# cache_control as it appears in Anthropic API (clawgate strips it, but we test
# whether having it in the payload changes anything before stripping)
def msgs_with_cache_control(system_body):
    # Simulate what Claude Code sends before DISABLE_PROMPT_CACHING
    # We embed it as a note in the system message (clawgate strips the field)
    return [
        {"role": "system", "content": system_body,
         "cache_control": {"type": "ephemeral"}},  # field ignored by llama-server
        *HISTORY,
    ]

# ---------------------------------------------------------------------------
# Test 1: Attribution header at realistic context size
# ---------------------------------------------------------------------------

def test1():
    hdr("TEST 1: Attribution header — realistic context (~50K tokens)")
    print("  Rounds: 3 per scenario. max_tokens=5 (prefill-only measurement).")
    print()

    import random, string
    def rand_cch(): return ''.join(random.choices('0123456789abcdef', k=5))

    small_sys  = BLOCK_500   # ~500 tokens
    large_sys  = BLOCK_50K   # ~50K tokens

    results = {}
    for label, body, use_header in [
        ("small (~500t) WITH header",    small_sys, True),
        ("small (~500t) WITHOUT header", small_sys, False),
        ("large (~50K) WITH header",     large_sys, True),
        ("large (~50K) WITHOUT header",  large_sys, False),
    ]:
        rates = []
        for _ in range(3):
            msgs = (msgs_with_header(body, rand_cch()) if use_header
                    else msgs_without_header(body))
            res = chat(msgs)
            rates.append(prefill_rate(res))
            print(f"    {label[:40]:<40} {rates[-1]:.0f} tok/s  ({res['prompt_tokens']} tokens, {res['elapsed_s']:.1f}s)")
        results[label] = statistics.mean(rates)

    print()
    print("  Summary:")
    for label, avg in results.items():
        print(f"    {label:<40} {avg:.0f} tok/s avg")

    small_gain = (results["small (~500t) WITHOUT header"] / results["small (~500t) WITH header"] - 1) * 100
    large_gain = (results["large (~50K) WITHOUT header"] / results["large (~50K) WITH header"] - 1) * 100
    print()
    print(f"  Gain (small context): +{small_gain:.0f}%")
    print(f"  Gain (large context): +{large_gain:.0f}%")
    return results

# ---------------------------------------------------------------------------
# Test 2: Flash attention
# ---------------------------------------------------------------------------

def test2_baseline():
    hdr("TEST 2: Flash attention — BASELINE (server as-is)")
    print("  Measuring prefill AND generation tok/s.")
    print()

    results = {"prefill": [], "gen": []}
    for size, body in [("small (~500t)", BLOCK_500), ("large (~50K)", BLOCK_50K)]:
        for i in range(2):
            msgs = msgs_without_header(body)
            res = chat(msgs, max_tokens=30)
            pt = res["prompt_tokens"]
            ct = res["completion_tokens"]
            elapsed = res["elapsed_s"]
            p_rate = prefill_rate(res)
            # Rough gen rate: subtract estimated prefill time
            p_time = pt / max(p_rate, 1)
            g_time = max(0.05, elapsed - p_time)
            g_rate = ct / g_time if g_time > 0 else 0
            print(f"    {size} round {i+1}: {pt}t prompt  prefill={p_rate:.0f} tok/s  gen={g_rate:.0f} tok/s")
            results["prefill"].append(p_rate)
            results["gen"].append(g_rate)

    print()
    print(f"  Avg prefill: {statistics.mean(results['prefill']):.0f} tok/s")
    print(f"  Avg gen:     {statistics.mean(results['gen']):.0f} tok/s")
    return results

def test2_fa():
    hdr("TEST 2: Flash attention — WITH -fa (restart server first!)")
    print("  Re-run after restarting llama-server with -fa flag.")
    print("  Add '-fa' to $cmd in run_server.ps1 and restart.")
    print()
    print("  Running measurements now (assumes server already restarted with -fa)...")
    print()
    return test2_baseline.__wrapped__() if hasattr(test2_baseline, '__wrapped__') else test2_baseline()

# ---------------------------------------------------------------------------
# Test 3: DISABLE_PROMPT_CACHING
# ---------------------------------------------------------------------------

def test3():
    hdr("TEST 3: DISABLE_PROMPT_CACHING — cache_control block overhead")
    print("  Claude Code adds cache_control:{type:ephemeral} to system prompt blocks.")
    print("  Clawgate strips these before forwarding to llama-server.")
    print("  Testing whether the extra JSON in the request pipeline adds latency.")
    print()

    # We test at the clawgate level: send Anthropic-format requests
    # with and without cache_control blocks and measure round-trip time.
    # Since this goes through clawgate (port 8082), it tests the translation overhead.
    CLAWGATE_URL = "http://127.0.0.1:8082"

    def anthropic_request(with_cache_control: bool):
        system = [{"type": "text", "text": BLOCK_500}]
        if with_cache_control:
            system[0]["cache_control"] = {"type": "ephemeral"}

        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 5,
            "system": system,
            "messages": [{"role": "user", "content": m["content"]}
                         for m in HISTORY if m["role"] == "user"][:1],
        }
        t0 = time.monotonic()
        r = httpx.post(f"{CLAWGATE_URL}/v1/messages",
                       json=payload,
                       headers={"x-api-key": "freecc",
                                "anthropic-version": "2023-06-01"},
                       timeout=120)
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        usage = r.json().get("usage", {})
        return {"elapsed_s": elapsed, "input_tokens": usage.get("input_tokens", 0)}

    # Check clawgate is reachable
    try:
        httpx.get(f"{CLAWGATE_URL}/v1/models", timeout=2)
    except Exception:
        print("  Clawgate not running on :8082 — skipping clawgate latency test.")
        print("  Testing llama-server directly instead (cache_control stripped before it arrives).")
        print()

        # Direct llama-server test: cache_control is just extra JSON in the field
        # llama-server ignores unknown fields, so expect no difference
        rates_with = []
        rates_without = []
        for i in range(3):
            r1 = chat(msgs_without_header(BLOCK_500))
            r2 = chat(msgs_with_cache_control(BLOCK_500))
            rates_without.append(prefill_rate(r1))
            rates_with.append(prefill_rate(r2))
            print(f"    Round {i+1}: without={rates_without[-1]:.0f}  with={rates_with[-1]:.0f} tok/s")

        avg_with    = statistics.mean(rates_with)
        avg_without = statistics.mean(rates_without)
        diff = (avg_without / avg_with - 1) * 100
        print()
        print(f"  Avg WITH cache_control:    {avg_with:.0f} tok/s")
        print(f"  Avg WITHOUT cache_control: {avg_without:.0f} tok/s")
        print(f"  Difference: {diff:+.1f}%  (expected ~0% — llama-server ignores the field)")
        return

    # Clawgate available — test full pipeline latency
    times_with    = []
    times_without = []
    print("  Testing via clawgate (:8082) — 3 rounds each:")
    for i in range(3):
        r1 = anthropic_request(with_cache_control=True)
        r2 = anthropic_request(with_cache_control=False)
        times_with.append(r1["elapsed_s"])
        times_without.append(r2["elapsed_s"])
        print(f"    Round {i+1}: with={r1['elapsed_s']:.2f}s  without={r2['elapsed_s']:.2f}s")

    avg_with    = statistics.mean(times_with)
    avg_without = statistics.mean(times_without)
    diff_ms = (avg_with - avg_without) * 1000
    print()
    print(f"  Avg WITH cache_control:    {avg_with:.3f}s")
    print(f"  Avg WITHOUT cache_control: {avg_without:.3f}s")
    print(f"  Difference: {diff_ms:+.1f}ms")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="all", choices=["1", "2", "3", "all"])
    args = p.parse_args()

    try:
        health_check()
    except Exception as e:
        print(f"llama-server not reachable at {LLAMA_URL}: {e}")
        return

    print("llama-server OK\n")

    # Warm up
    print("Warming up (1 request)...", end=" ", flush=True)
    chat(msgs_without_header(BLOCK_500))
    print("done\n")

    if args.test in ("1", "all"):
        test1()
    if args.test in ("2", "all"):
        test2_baseline()
        print()
        print("  *** To test flash attention: ***")
        print("  1. Add '-fa' to $cmd array in bench/local/run_server.ps1")
        print("  2. Restart the server")
        print("  3. Run: python bench/test_perf_options.py --test 2")
    if args.test in ("3", "all"):
        test3()

    print("\nDone.")

if __name__ == "__main__":
    main()
