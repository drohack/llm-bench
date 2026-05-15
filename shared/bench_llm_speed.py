import argparse
import csv
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Optional

from openai import OpenAI

API_KEY = "your-nvidia-api-key"
BASE_URL = "https://integrate.api.nvidia.com/v1"

try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    if ENCODING is not None:
        return len(ENCODING.encode(text))
    return max(1, len(text) // 4)


def make_prompt(target_prompt_tokens: int) -> str:
    seed = """
You are debugging a TypeScript full-stack web app. Analyze the code and explain a careful fix plan.

File: src/app/lib/anilist.ts

export async function getSeasonalAnime(season: string, year: number) {
  const response = await fetch('/api/anilist', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ season, year }),
  });

  if (!response.ok) {
    throw new Error('Failed to load anime data');
  }

  const data = await response.json();
  return data.Page.media;
}

Task:
Explain how you would diagnose a bug where the homepage remains stuck on "Loading seasonal anime..."
even though the backend endpoint returns JSON. Include likely causes, checks, and a minimal fix.
"""
    chunks = []
    while approx_tokens("\n".join(chunks)) < target_prompt_tokens:
        chunks.append(seed)
    return "\n".join(chunks)


@dataclass
class BenchResult:
    model: str
    run: int
    prompt_tokens_est: int
    completion_tokens_est: int
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    total_seconds: float
    ttft_seconds: Optional[float]
    prefill_tok_per_sec: Optional[float]   # prompt tokens / TTFT -- the real bottleneck for long prompts
    tokens_per_second_est: Optional[float]
    tokens_per_second_usage: Optional[float]
    chars_per_second: float
    output_chars: int
    error: str = ""


def stream_once(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    run: int,
    thinking: bool = False,
    fixture_messages: Optional[list] = None,
    fixture_tools: Optional[list] = None,
    gen_prompt: Optional[str] = None,   # if set, replaces last user msg in fixture for gen benchmarking
    attribution_header: bool = False,   # prepend changing cch billing header to simulate default Claude Code behavior
) -> BenchResult:
    start = time.perf_counter()
    first_token_at = None
    output_parts = []
    usage_prompt_tokens = None
    usage_completion_tokens = None

    if fixture_messages is not None:
        messages = list(fixture_messages)  # copy
        if attribution_header:
            # Prepend a billing header with a unique cch hash per request, simulating
            # current Claude Code behavior (CLAUDE_CODE_ATTRIBUTION_HEADER not disabled).
            # This invalidates llama-server's KV cache prefix on every turn.
            cch = ''.join(random.choices('0123456789abcdef', k=5))
            header_msg = {"role": "system",
                          "content": f"x-anthropic-billing-header: cc_version=2.1.136; cc_entrypoint=claude-vscode; cch={cch};"}
            messages = [header_msg] + messages
        if gen_prompt:
            # Replace the last user message so the model generates long text instead of a tool call.
            # All prior messages (the huge system prefix) are identical so KV cache stays warm.
            for i in reversed(range(len(messages))):
                if messages[i].get("role") == "user":
                    messages[i] = {"role": "user", "content": gen_prompt}
                    break
    else:
        messages = [
            {"role": "system", "content": "You are a concise senior software engineer. Answer directly."},
            {"role": "user",   "content": prompt},
        ]

    # Don't pass tools when using gen_prompt -- we want text output, not tool calls.
    tools = (fixture_tools if fixture_tools and not gen_prompt else None)

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    if tools:
        kwargs["tools"] = tools
    if thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}

    # Some OpenAI-compatible endpoints support this; some reject it.
    # Try it first so we can get server-reported token counts.
    try:
        stream = client.chat.completions.create(
            **kwargs,
            stream_options={"include_usage": True},
        )
    except Exception:
        stream = client.chat.completions.create(**kwargs)

    reasoning_parts = []
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage_prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
            usage_completion_tokens = getattr(chunk.usage, "completion_tokens", None)

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        text = getattr(delta, "content", None)

        # Track TTFT on first token of any kind (reasoning, content, or tool_call)
        tool_calls = getattr(delta, "tool_calls", None)
        if (reasoning or text or tool_calls) and first_token_at is None:
            first_token_at = time.perf_counter()

        if reasoning:
            reasoning_parts.append(reasoning)
        if text:
            output_parts.append(text)

    end = time.perf_counter()
    output = "".join(output_parts)
    reasoning = "".join(reasoning_parts)
    all_output = output + reasoning  # count all generated text for speed metrics

    prompt_tokens_est = approx_tokens(prompt) if fixture_messages is None else 0
    completion_tokens_est = approx_tokens(all_output)
    total_seconds = end - start
    ttft_seconds = None if first_token_at is None else first_token_at - start

    prefill_tok_per_sec = None
    if ttft_seconds and ttft_seconds > 0 and usage_prompt_tokens:
        prefill_tok_per_sec = usage_prompt_tokens / ttft_seconds

    gen_seconds = None
    if first_token_at is not None:
        gen_seconds = max(0.000001, end - first_token_at)

    tokens_per_second_est = None
    if gen_seconds and completion_tokens_est > 0:
        tokens_per_second_est = completion_tokens_est / gen_seconds

    tokens_per_second_usage = None
    if gen_seconds and usage_completion_tokens:
        tokens_per_second_usage = usage_completion_tokens / gen_seconds

    chars_per_second = len(all_output) / max(0.000001, total_seconds)

    return BenchResult(
        model=model,
        run=run,
        prompt_tokens_est=prompt_tokens_est,
        completion_tokens_est=completion_tokens_est,
        usage_prompt_tokens=usage_prompt_tokens,
        usage_completion_tokens=usage_completion_tokens,
        total_seconds=total_seconds,
        ttft_seconds=ttft_seconds,
        prefill_tok_per_sec=prefill_tok_per_sec,
        tokens_per_second_est=tokens_per_second_est,
        tokens_per_second_usage=tokens_per_second_usage,
        chars_per_second=chars_per_second,
        output_chars=len(all_output),
    )


def summarize(results: list[BenchResult]) -> None:
    good = [r for r in results if not r.error]
    if not good:
        print("No successful runs.")
        return

    def vals(attr: str):
        return [getattr(r, attr) for r in good if getattr(r, attr) is not None]

    def stat_line(label: str, attr: str, unit: str = ""):
        v = vals(attr)
        if not v:
            print(f"{label}: n/a")
            return
        print(
            f"{label}: avg={statistics.mean(v):.3f}{unit}, "
            f"min={min(v):.3f}{unit}, "
            f"max={max(v):.3f}{unit}"
        )

    print("\n=== SUMMARY ===")
    print(f"model: {good[0].model}")
    print(f"successful runs: {len(good)} / {len(results)}")
    stat_line("TTFT (prefill)", "ttft_seconds", "s")
    stat_line("prefill tok/s", "prefill_tok_per_sec", "")
    stat_line("total latency", "total_seconds", "s")
    stat_line("est output tok/sec after first token", "tokens_per_second_est", "")
    stat_line("usage output tok/sec after first token", "tokens_per_second_usage", "")
    stat_line("chars/sec total", "chars_per_second", "")
    stat_line("prompt tokens (server)", "usage_prompt_tokens", "")
    stat_line("completion tokens (server)", "usage_completion_tokens", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=BASE_URL or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=API_KEY or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (GLM4.7 etc.)")
    parser.add_argument("--fixture", default=None,
                        help="Path to fixture_real_request.json -- sends the real Claude Code request "
                             "instead of a synthetic prompt. Reports prefill_tok_per_sec.")
    parser.add_argument("--warmup", action="store_true",
                        help="Do one untimed warmup run with the fixture before timed runs (primes KV cache).")
    parser.add_argument("--gen-prompt", default=None,
                        help="Replace the last user message in the fixture with this text for gen tok/s measurement. "
                             "Use with --warmup so the prefix cache is warm before the timed runs.")
    parser.add_argument("--attribution-header", action="store_true",
                        help="Inject a changing x-anthropic-billing-header as the first system message each run, "
                             "simulating default Claude Code behavior (CLAUDE_CODE_ATTRIBUTION_HEADER not set). "
                             "Compare runs with and without to measure the KV cache invalidation cost.")
    parser.add_argument("--csv", default="results/speed_results.csv")
    args = parser.parse_args()

    if not args.base_url:
        raise SystemExit("Missing --base-url or OPENAI_BASE_URL")
    if not args.api_key:
        raise SystemExit("Missing --api-key or OPENAI_API_KEY")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    fixture_messages = None
    fixture_tools = None
    fixture_messages = None
    fixture_tools    = None
    gen_prompt       = args.gen_prompt

    if args.fixture:
        with open(args.fixture, encoding="utf-8") as f:
            fix = json.load(f)
        fixture_messages = fix.get("messages", [])
        fixture_tools    = fix.get("tools") or None
        prompt = ""
        mode = "gen-prompt" if gen_prompt else "tool-call"
        print(f"Benchmarking model: {args.model}  [fixture/{mode}]")
        print(f"Fixture: {args.fixture}  messages={len(fixture_messages)}  tools={len(fixture_tools or [])}")
        if gen_prompt:
            print(f"Gen prompt: {gen_prompt[:80]}...")
    else:
        prompt = make_prompt(args.prompt_tokens)
        print(f"Benchmarking model: {args.model}")
        print(f"Estimated prompt tokens: {approx_tokens(prompt)}")

    results: list[BenchResult] = []
    print(f"Base URL: {args.base_url}")
    print(f"Runs: {args.runs}  Max output tokens: {args.max_tokens}")

    if args.attribution_header:
        print("Attribution header: ON (changing cch per request — simulates default Claude Code)")
    else:
        print("Attribution header: OFF (CLAUDE_CODE_ATTRIBUTION_HEADER=0 behavior)")

    if args.warmup and fixture_messages is not None:
        print("\nWarmup run (priming KV cache, not timed)...", flush=True)
        try:
            stream_once(client=client, model=args.model, prompt="", max_tokens=50,
                        temperature=args.temperature, run=0,
                        fixture_messages=fixture_messages, fixture_tools=fixture_tools)
            print("  Cache primed.")
        except Exception as e:
            print(f"  Warmup failed (continuing anyway): {e}")
    print()

    for i in range(1, args.runs + 1):
        print(f"Run {i}/{args.runs}...", flush=True)
        try:
            r = stream_once(
                client=client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                run=i,
                thinking=args.thinking,
                fixture_messages=fixture_messages,
                fixture_tools=fixture_tools,
                gen_prompt=gen_prompt,
                attribution_header=args.attribution_header,
            )
            results.append(r)
            ttft_str    = f"{r.ttft_seconds:.1f}s"    if r.ttft_seconds    is not None else "n/a"
            prefill_str = f"{r.prefill_tok_per_sec:.0f} tok/s" if r.prefill_tok_per_sec is not None else "n/a"
            gen_str     = f"{r.tokens_per_second_usage:.1f}" if r.tokens_per_second_usage is not None else "n/a"
            print(f"  TTFT={ttft_str}  prefill={prefill_str}  gen={gen_str} tok/s  total={r.total_seconds:.1f}s")
        except Exception as e:
            r = BenchResult(
                model=args.model,
                run=i,
                prompt_tokens_est=approx_tokens(prompt),
                completion_tokens_est=0,
                usage_prompt_tokens=None,
                usage_completion_tokens=None,
                total_seconds=0.0,
                ttft_seconds=None,
                prefill_tok_per_sec=None,
                tokens_per_second_est=None,
                tokens_per_second_usage=None,
                chars_per_second=0.0,
                output_chars=0,
                error=repr(e),
            )
            results.append(r)
            print(f"  ERROR: {r.error}")

    summarize(results)

    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        if f.tell() == 0:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nWrote results to: {args.csv}")


if __name__ == "__main__":
    main()
