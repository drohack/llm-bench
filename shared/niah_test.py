"""
Needle in a Haystack (NIAH) context window test.

Hides a secret passcode at various depths in a long filler document and checks
whether the model can retrieve it. Tests whether long context actually works.

Usage:
    python niah_test.py --model my-model --base-url http://127.0.0.1:8081/v1 --api-key none
    python niah_test.py --provider nvidia --model qwen/qwen3.5-122b-a10b
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI

NEEDLE   = "The secret passcode is: VIOLET-7734"
QUESTION = "What is the secret passcode mentioned in the document? Reply with only the passcode, nothing else."
FILLER   = (
    "The development of artificial intelligence has a rich history spanning several decades. "
    "Researchers have explored many approaches to machine learning, natural language processing, "
    "computer vision, and reasoning systems. Early symbolic AI gave way to statistical methods, "
    "and later to deep neural networks trained on large datasets. Each generation of models "
    "brought new capabilities while also revealing new limitations and challenges. "
)

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY  = os.environ.get("NVIDIA_API_KEY", "your-nvidia-api-key")


def make_context(approx_tokens: int, depth_pct: float) -> str:
    """Build a filler document with the needle inserted at depth_pct% through it."""
    chars_needed = approx_tokens * 4  # rough chars-per-token estimate
    filler_block = FILLER * (chars_needed // len(FILLER) + 1)
    filler_block = filler_block[:chars_needed]
    insert = int(len(filler_block) * depth_pct / 100)
    return filler_block[:insert] + f"\n\n{NEEDLE}\n\n" + filler_block[insert:]


@dataclass
class NIAHResult:
    model: str
    context_k_tokens: int
    depth_pct: int
    passed: bool
    answer: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    gen_tok_per_sec: float   # output tok/s after first token (excludes prefill)
    prefill_tok_per_sec: float  # prompt tokens / TTFT
    error: str = ""


def run_one(client: OpenAI, model: str, context_k: int, depth_pct: int,
            timeout: int) -> NIAHResult:
    context = make_context(context_k * 1000, depth_pct)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Read the document carefully and answer questions about it."
        },
        {
            "role": "user",
            "content": f"Document:\n{context}\n\nQuestion: {QUESTION}"
        }
    ]
    start = time.perf_counter()
    first_token_at = None
    output_parts = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=30,
            temperature=0,
            timeout=timeout,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                prompt_tokens     = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            text = getattr(chunk.choices[0].delta, "content", None)
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_parts.append(text)

        elapsed   = time.perf_counter() - start
        ttft      = (first_token_at - start) if first_token_at else elapsed
        gen_secs  = max(0.000001, elapsed - ttft)
        answer    = "".join(output_parts).strip()
        passed    = "VIOLET-7734" in answer

        gen_tok_s    = completion_tokens / gen_secs if completion_tokens else 0.0
        prefill_tok_s = prompt_tokens / ttft if (prompt_tokens and ttft > 0) else 0.0

        return NIAHResult(model=model, context_k_tokens=context_k,
                          depth_pct=depth_pct, passed=passed,
                          answer=answer[:80], latency_s=round(elapsed, 2),
                          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                          gen_tok_per_sec=round(gen_tok_s, 1),
                          prefill_tok_per_sec=round(prefill_tok_s, 1))
    except Exception as e:
        elapsed = time.perf_counter() - start
        return NIAHResult(model=model, context_k_tokens=context_k,
                          depth_pct=depth_pct, passed=False,
                          answer="", latency_s=round(elapsed, 2),
                          prompt_tokens=0, completion_tokens=0,
                          gen_tok_per_sec=0.0, prefill_tok_per_sec=0.0,
                          error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True)
    ap.add_argument("--base-url",  default=None)
    ap.add_argument("--api-key",   default=None)
    ap.add_argument("--provider",  choices=["local", "nvidia"], default="local")
    # Context sizes in thousands of tokens to test
    ap.add_argument("--context-sizes", nargs="+", type=int, default=[8, 32, 64, 100])
    # Depths (% through document) to test
    ap.add_argument("--depths",    nargs="+", type=int, default=[10, 50, 90])
    ap.add_argument("--timeout",   type=int, default=600)
    ap.add_argument("--csv",       default="results/niah_results.csv")
    args = ap.parse_args()

    if args.provider == "nvidia":
        base_url = args.base_url or NVIDIA_BASE
        api_key  = args.api_key  or NVIDIA_KEY
    else:
        base_url = args.base_url or "http://127.0.0.1:8081/v1"
        api_key  = args.api_key  or "none"

    client = OpenAI(api_key=api_key, base_url=base_url)

    results = []
    total = len(args.context_sizes) * len(args.depths)
    idx = 0

    print(f"Model: {args.model}")
    print(f"Tests: {len(args.context_sizes)} context sizes x {len(args.depths)} depths = {total} total")
    print(f"{'ctx_k':>6}  {'depth%':>6}  {'pass':>5}  {'latency':>8}  {'prefill t/s':>11}  {'gen t/s':>8}  answer")
    print("-" * 90)

    for ctx_k in args.context_sizes:
        for depth in args.depths:
            idx += 1
            r = run_one(client, args.model, ctx_k, depth, args.timeout)
            results.append(r)
            status = "PASS  " if r.passed else "FAIL  "
            print(f"{ctx_k:>5}K  {depth:>5}%  {status}  {r.latency_s:>7.1f}s  "
                  f"{r.prefill_tok_per_sec:>11.0f}  {r.gen_tok_per_sec:>8.1f}  {r.answer!r}")
            if r.error:
                print(f"         ERROR: {r.error}")

    passed = sum(1 for r in results if r.passed)
    print(f"\nScore: {passed}/{total}  ({100*passed//total}%)")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        if f.tell() == 0:
            w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"Results -> {args.csv}")


if __name__ == "__main__":
    main()
