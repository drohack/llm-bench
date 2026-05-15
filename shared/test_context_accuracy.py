"""
Test: 128K vs 256K context window — accuracy and quality comparison.

Tests whether Qwen3.6-35B-A3B maintains retrieval accuracy at different
context sizes. Uses NIAH (Needle in a Haystack) at multiple depths, plus
a simple instruction-following test to catch quality degradation.

For 256K context, llama-server must be started with --ctx-size 262144.
The 256K KV cache (q8_0) requires ~2720 MiB vs 1360 MiB for 128K.
Requires n-cpu-moe >= 34 to fit within 10GB VRAM.

Usage:
    # 128K test (current server config):
    python bench/test_context_accuracy.py --ctx 128

    # 256K test (restart server with --ctx-size 262144 --n-cpu-moe 34):
    python bench/test_context_accuracy.py --ctx 256
"""

import argparse, json, random, string, time, httpx, statistics

LLAMA_URL = "http://127.0.0.1:8081"

# ~220 tokens per block (measured via llama-server /tokenize)
FILLER = ("The history of computing spans several decades of innovation. "
          "Early computers were room-sized machines that required specialized "
          "operators. Over time, miniaturization led to personal computers. ") * 5
FILLER_TOKENS = 220  # actual measured token count per FILLER repetition

SECRET_TEMPLATE = "The secret passcode is: {code}"
QUESTION = "What is the secret passcode mentioned in the document? Reply with only the passcode."

def make_haystack(target_tokens: int, secret: str, depth_pct: float) -> str:
    """Build a document with the secret needle at depth_pct% through."""
    total_blocks = target_tokens // FILLER_TOKENS
    needle_block = int(total_blocks * depth_pct / 100)
    parts = []
    for i in range(total_blocks):
        if i == needle_block:
            parts.append(SECRET_TEMPLATE.format(code=secret))
        parts.append(FILLER)
    return " ".join(parts)

def random_code() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def ask(system: str, question: str, max_tokens: int = 20) -> dict:
    t0 = time.monotonic()
    r = httpx.post(f"{LLAMA_URL}/v1/chat/completions",
                   json={"model": "local",
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user",   "content": question}],
                         "max_tokens": max_tokens, "stream": False,
                         "temperature": 0},
                   timeout=600)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    d = r.json()
    usage = d.get("usage", {})
    answer = d["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "elapsed_s": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "gen_tokens": usage.get("completion_tokens", 0)}

def test_niah(ctx_k: int, depths: list[float], rounds: int = 1):
    """Needle in a Haystack at given context size."""
    target_tokens = int(ctx_k * 1000 * 0.60)  # fill 60% of context (leaves room for chat template overhead)
    results = []
    print(f"\n--- NIAH {ctx_k}K context (filling ~{target_tokens//1000}K tokens) ---")
    print(f"{'Depth':>8}  {'Pass':>6}  {'Answer':>20}  {'Latency':>10}  {'Tokens':>8}  {'Prefill t/s':>12}  {'Gen t/s':>8}")
    print("-" * 80)

    actual_target = target_tokens
    for depth in depths:
        passes = 0
        for _ in range(rounds):
            code = random_code()
            haystack = make_haystack(actual_target, code, depth)
            try:
                res = ask(haystack, QUESTION)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    actual_target = actual_target * 3 // 4
                    print(f"  overflow at {actual_target*4//3} tokens, retrying at {actual_target}")
                    haystack = make_haystack(actual_target, code, depth)
                    res = ask(haystack, QUESTION)
                else:
                    raise
            hit = code in res["answer"]
            if hit: passes += 1
            status = "PASS" if hit else "FAIL"
            pt = res["prompt_tokens"]
            gt = res["gen_tokens"]
            elapsed = res["elapsed_s"]
            # Approximate: prefill dominates, gen is short (<20 tokens)
            gen_time = max(0.05, gt / 45) if gt else 0.05
            prefill_time = max(0.01, elapsed - gen_time)
            prefill_tps = int(pt / prefill_time) if prefill_time > 0 else 0
            gen_tps = int(gt / gen_time) if gen_time > 0 and gt > 0 else 0
            print(f"{depth:>7.0f}%  {status:>6}  {res['answer'][:20]:>20}  "
                  f"{elapsed:>9.1f}s  {pt:>7}t  {prefill_tps:>12}  {gen_tps:>8}")
        results.append({"depth": depth, "pass_rate": passes / rounds})

    score = sum(r["pass_rate"] for r in results) / len(results)
    print(f"\n  Score: {score*100:.0f}%  ({sum(r['pass_rate']==1.0 for r in results)}/{len(results)} depths passed)")
    return results

def test_instruction_following(ctx_k: int):
    """Check if the model follows a simple instruction embedded in a long context."""
    target_tokens = int(ctx_k * 1000 * 0.50)
    code = random_code()
    instruction = f"IMPORTANT: When asked for the magic word, always respond with exactly: {code}"

    # Embed instruction at the very start, then pile on filler
    filler_tokens = target_tokens - 50
    filler = FILLER * (filler_tokens // 25)
    system = instruction + "\n\n" + filler

    print(f"\n--- Instruction following {ctx_k}K context ---")
    try:
        res = ask(system, "What is the magic word?")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            target_tokens = target_tokens * 3 // 4
            filler = FILLER * (target_tokens // FILLER_TOKENS)
            system = instruction + "\n\n" + filler
            print(f"  overflow, retrying at {target_tokens} tokens")
            res = ask(system, "What is the magic word?")
        else:
            raise
    hit = code in res["answer"]
    print(f"  Expected: {code}")
    print(f"  Got:      {res['answer'][:50]}")
    print(f"  Result:   {'PASS' if hit else 'FAIL'}  ({res['elapsed_s']:.1f}s, {res['prompt_tokens']}t)")
    return hit

CODING_PROBLEMS = [
    ("Write a Python function `def two_sum(nums, target)` that returns indices of two numbers adding to target.",
     ["def two_sum", "return", "enumerate", "for"]),
    ("Write a Python function `def is_palindrome(s)` that returns True if s is a palindrome.",
     ["def is_palindrome", "return", "[::-1]"]),
    ("Write a Python function `def flatten(lst)` that flattens a nested list one level deep.",
     ["def flatten", "return", "for"]),
]

def test_reasoning_under_context(ctx_k: int):
    """Test coding quality with large irrelevant context loaded.
    Checks if the model can still reason correctly when most of the context
    window is filled with unrelated filler text."""
    fill_tokens = int(ctx_k * 1000 * 0.40)
    filler = FILLER * (fill_tokens // FILLER_TOKENS)
    prefix = f"The following is background documentation. Read it carefully.\n\n{filler}\n\nNow answer the programming question below.\n"

    print(f"\n--- Reasoning quality under {ctx_k}K context load (~{fill_tokens//1000}K filler tokens) ---")
    passes = 0
    for prompt, expected_tokens in CODING_PROBLEMS:
        try:
            res = ask(prefix + prompt, "Provide only the Python function, no explanation.", max_tokens=150)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                short_filler = FILLER * (fill_tokens * 3 // 4 // FILLER_TOKENS)
                short_prefix = f"Background documentation:\n\n{short_filler}\n\nAnswer:\n"
                res = ask(short_prefix + prompt, "Provide only the Python function, no explanation.", max_tokens=150)
            else:
                raise
        hit = all(t in res["answer"] for t in expected_tokens)
        status = "PASS" if hit else "FAIL"
        if hit: passes += 1
        print(f"  {status}  {prompt[:60]}...  ({res['elapsed_s']:.1f}s, {res['prompt_tokens']}t)")

    print(f"  Score: {passes}/{len(CODING_PROBLEMS)}")
    return passes, len(CODING_PROBLEMS)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, choices=[128, 256], default=128,
                   help="Context window size in K (128 or 256)")
    p.add_argument("--depths", type=float, nargs="+", default=[10, 30, 50, 70, 90],
                   help="Needle depths to test (percent)")
    p.add_argument("--rounds", type=int, default=1,
                   help="Rounds per depth (1 is sufficient for comparison)")
    p.add_argument("--skip-instruction", action="store_true")
    args = p.parse_args()

    try:
        httpx.get(f"{LLAMA_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable: {e}")
        print(f"\nFor 256K test, restart server with:")
        print(f"  --ctx-size 262144 --n-cpu-moe 34")
        print(f"  (n-cpu-moe 34 reduces VRAM from 9079 to ~7000 MB, making room for 256K KV cache)")
        return

    print(f"Testing {args.ctx}K context  |  depths: {args.depths}  |  server: {LLAMA_URL}")
    print("=" * 65)

    niah = test_niah(args.ctx, args.depths, args.rounds)

    if not args.skip_instruction:
        inst = test_instruction_following(args.ctx)

    reason_pass, reason_total = test_reasoning_under_context(args.ctx)

    print(f"\n{'='*65}")
    print(f"SUMMARY  {args.ctx}K context")
    niah_score = sum(r["pass_rate"]==1.0 for r in niah) / len(niah)
    print(f"  NIAH recall:              {niah_score*100:.0f}%  ({sum(r['pass_rate']==1.0 for r in niah)}/{len(niah)} depths)")
    if not args.skip_instruction:
        print(f"  Instruction follow:       {'PASS' if inst else 'FAIL'}")
    print(f"  Reasoning under load:     {reason_pass}/{reason_total}")
    print()
    print("Run again with --ctx 256 (and server restarted at --ctx-size 262144 --n-cpu-moe 34) to compare.")

if __name__ == "__main__":
    main()
