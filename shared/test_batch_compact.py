"""
Benchmark: batch size impact on compact-like cold prefill speed.

A compact request is a cold re-prefill of the full context (~90-111K tokens).
This test simulates that by sending a large prompt and measuring TTFT.

Tests b=8192 (current) vs b=16384 side-by-side.

Usage:
  1. Start server: .\bench\local\run_server.ps1 (with --n-cpu-moe 26 for VRAM headroom)
  2. Run:  python bench/test_batch_compact.py
  3. Then change -b 16384 in run_server.ps1 and restart server
  4. Run again to compare

Or run with --batch 16384 if server is already running with that batch size.
"""

import argparse, json, time, httpx, statistics

LLAMA_URL = "http://127.0.0.1:8081"

# Realistic Claude Code system prompt size for comparison points
BLOCK = ("You are Claude Code, Anthropic's official CLI for Claude, "
         "running within the Claude Agent SDK. ") * 60  # ~500 tokens

def make_large_context(target_k: int) -> list[dict]:
    """Build a prompt of roughly target_k thousand tokens."""
    repeats = (target_k * 1000) // 500
    body = BLOCK * repeats
    return [{"role": "system", "content": body},
            {"role": "user",   "content": "Summarize the above context briefly."}]

def send(messages: list[dict], max_tokens: int = 10) -> dict:
    t0 = time.monotonic()
    r = httpx.post(f"{LLAMA_URL}/v1/chat/completions",
                   json={"model": "local", "messages": messages,
                         "max_tokens": max_tokens, "stream": False},
                   timeout=600)
    elapsed = time.monotonic() - t0
    if r.status_code == 400:
        raise OverflowError(f"Prompt too large for context window (400): {r.text[:100]}")
    r.raise_for_status()
    usage = r.json().get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    return {"elapsed_s": elapsed, "prompt_tokens": pt,
            "prefill_tok_s": pt / elapsed if elapsed > 0 else 0}

def run_scenario(label: str, context_k: int, rounds: int = 3):
    print(f"\n{'='*64}")
    print(f"  {label}  (~{context_k}K token cold prefill)")
    print(f"{'='*64}")
    msgs = make_large_context(context_k)

    # Cold run first (no cache)
    rates = []
    import random, string
    actual_k = context_k
    for i in range(rounds):
        # Unique suffix ensures cold prefill each round
        msgs = make_large_context(actual_k)
        msgs[0]["content"] += ''.join(random.choices(string.ascii_letters, k=10))
        try:
            res = send(msgs)
        except OverflowError:
            actual_k = actual_k * 3 // 4
            print(f"overflow — retrying at {actual_k}K target", end=" ... ", flush=True)
            msgs = make_large_context(actual_k)
            msgs[0]["content"] += ''.join(random.choices(string.ascii_letters, k=10))
            res = send(msgs)
        print(f"  Round {i+1}: {res['prompt_tokens']}t  "
              f"TTFT={res['elapsed_s']:.1f}s  "
              f"prefill={res['prefill_tok_s']:.0f} tok/s")
        rates.append(res["prefill_tok_s"])
        times = [res["elapsed_s"]]

    avg_rate = statistics.mean(rates)
    avg_time = statistics.mean(times)
    print(f"\n  Average: {avg_rate:.0f} tok/s  ({avg_time:.1f}s TTFT)")
    return avg_rate, avg_time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=None,
                   help="Current server batch size (for labeling only)")
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    try:
        httpx.get(f"{LLAMA_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable: {e}")
        return

    # Get actual server batch size from slots endpoint
    batch_label = f"b={args.batch}" if args.batch else "b=unknown (check server flags)"
    print(f"Server: {LLAMA_URL}  Batch size: {batch_label}")
    print()
    print("Simulating compact-scale cold prefills at different context sizes.")
    print("Each round uses a unique prompt to prevent cache reuse.")

    results = {}
    for ctx_k in [50]:  # generates ~120K actual tokens (compact-scale cold prefill)
        label = f"{batch_label}  {ctx_k}K context"
        rate, ttft = run_scenario(label, ctx_k, args.rounds)
        results[f"{ctx_k}k"] = {"tok_s": rate, "ttft_s": ttft}

    print(f"\n{'='*64}")
    print(f"SUMMARY  ({batch_label})")
    print(f"{'='*64}")
    for k, v in results.items():
        print(f"  {k}: {v['tok_s']:.0f} tok/s avg  ({v['ttft_s']:.1f}s TTFT)")
    print()
    print("Compare against run with different -b value.")
    print("Expected compact context: ~90-111K tokens at 85% of 131K window.")

if __name__ == "__main__":
    main()
