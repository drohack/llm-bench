"""
Benchmark: impact of CLAUDE_CODE_ATTRIBUTION_HEADER on KV cache reuse.

Claude Code injects a billing header as the first system prompt block:
  "x-anthropic-billing-header: cc_version=...; cch=<unique_hash>"
The cch value changes on nearly every request, so llama-server's LCP prefix
matching finds zero common prefix and re-prefills the full context each turn.

This test sends two back-to-back requests that share a long common prefix,
once WITH the changing header (simulating current behavior) and once WITHOUT.
We measure prefill tokens/sec and LCP similarity from llama-server's response.

Usage:
    python bench/test_attribution_header.py
"""

import json, time, httpx, statistics

LLAMA_URL = "http://127.0.0.1:8081"

# Realistic Claude Code system prompt size: ~3K tokens of text
SYSTEM_BODY = "You are Claude Code, Anthropic's official CLI for Claude.\n" * 120

# A few turns of conversation history (~1K tokens)
HISTORY = [
    {"role": "user",      "content": "Read src/index.ts and summarize it."},
    {"role": "assistant", "content": "The file exports an Express app with three routes: /health, /api/anilist, and /api/seasons. It uses a middleware stack including rate limiting and error handling."},
    {"role": "user",      "content": "What does the rate limiter do?"},
    {"role": "assistant", "content": "The rate limiter uses express-rate-limit to cap requests at 100 per 15 minutes per IP address. It returns a 429 JSON response when exceeded."},
]

NEW_TURN = {"role": "user", "content": "Show me the seasons route handler."}


def make_messages(with_header: bool, cch_value: str) -> list[dict]:
    system_blocks = []
    if with_header:
        system_blocks.append({
            "role": "system",
            "content": f"x-anthropic-billing-header: cc_version=2.1.136; cc_entrypoint=claude-vscode; cch={cch_value};"
        })
    system_blocks.append({"role": "system", "content": SYSTEM_BODY})
    # Flatten: llama-server OpenAI format uses messages array, system is role=system
    msgs = [{"role": "system", "content": SYSTEM_BODY}]
    if with_header:
        # Prepend header as first system message
        msgs = [
            {"role": "system", "content": f"x-anthropic-billing-header: cc_version=2.1.136; cc_entrypoint=claude-vscode; cch={cch_value};"},
            {"role": "system", "content": SYSTEM_BODY},
        ]
    msgs += HISTORY + [NEW_TURN]
    return msgs


def send_request(messages: list[dict]) -> dict:
    """Send chat completion request, return timing_info from response."""
    payload = {
        "model": "local",
        "messages": messages,
        "max_tokens": 5,  # We only care about prefill, not generation
        "stream": False,
    }
    t0 = time.monotonic()
    r = httpx.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
    }


def get_last_slot_stats() -> dict | None:
    """Pull timing info from llama-server /slots endpoint."""
    try:
        r = httpx.get(f"{LLAMA_URL}/slots", timeout=5)
        slots = r.json()
        if slots:
            return slots[0]
    except Exception:
        pass
    return None


def run_scenario(label: str, with_header: bool, n_rounds: int = 3):
    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print(f"{'='*60}")

    prefill_rates = []
    import random, string

    for i in range(n_rounds):
        # Each round: simulate a new turn (header cch changes if with_header=True)
        cch = ''.join(random.choices(string.hexdigits[:16], k=5)) if with_header else "00000"
        msgs = make_messages(with_header=with_header, cch_value=cch)

        print(f"  Round {i+1}: cch={'changing' if with_header else 'absent'} ({cch})", end=" ... ", flush=True)
        result = send_request(msgs)

        pt = result["prompt_tokens"]
        elapsed = result["elapsed_s"]
        rate = pt / elapsed if elapsed > 0 else 0
        prefill_rates.append(rate)
        print(f"{pt} prompt tokens in {elapsed:.1f}s = {rate:.0f} tok/s prefill")

    avg = statistics.mean(prefill_rates)
    print(f"\n  Average prefill: {avg:.0f} tok/s")
    return avg


def main():
    # Warm up: send one request so the server is ready
    print("Warming up llama-server...")
    try:
        send_request(make_messages(with_header=False, cch_value="00000"))
        print("OK\n")
    except Exception as e:
        print(f"FAILED: {e}")
        print(f"Make sure llama-server is running on {LLAMA_URL}")
        return

    # Run with changing header (current Claude Code behavior)
    rate_with = run_scenario("WITH attribution header (current behavior)", with_header=True, n_rounds=3)

    # Small pause to let server settle
    time.sleep(2)

    # Run without header (CLAUDE_CODE_ATTRIBUTION_HEADER=0)
    rate_without = run_scenario("WITHOUT attribution header (CLAUDE_CODE_ATTRIBUTION_HEADER=0)", with_header=False, n_rounds=3)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  With header:    {rate_with:.0f} tok/s avg prefill")
    print(f"  Without header: {rate_without:.0f} tok/s avg prefill")
    if rate_with > 0:
        improvement = (rate_without - rate_with) / rate_with * 100
        print(f"  Improvement:    {improvement:+.1f}%")
    print()
    print("Note: improvement shows up as higher prefill tok/s (faster time-to-first-token).")
    print("The effect grows with context size — longer sessions benefit more.")


if __name__ == "__main__":
    main()
