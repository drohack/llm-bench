"""
Test whether models call tools correctly using the real production payload
captured from a clawgate debug log (full 50+ tool list, real system prompt).

Usage:
    python test_real_tools.py                        # uses fixture_real_request.json
    python test_real_tools.py --fixture my.json      # custom fixture
"""
import argparse
import csv
import json
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

import httpx

PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.environ.get("NVIDIA_API_KEY", "your-nvidia-api-key"),
        "models": [
            "deepseek-ai/deepseek-v4-pro",
            "z-ai/glm-5.1",
            "z-ai/glm5",
            "z-ai/glm4.7",
            "qwen/qwen3.5-397b-a17b",
        ],
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": "csk-pth3t28e3x38trcnm5t46xpxnmk8cwvfen4k2p6d4x2vpc3x",
        "models": [
            "zai-glm-4.7",
            "qwen-3-235b-a22b-instruct-2507",
            "gpt-oss-120b",
        ],
    },
    "local": {
        "base_url": "http://127.0.0.1:8081/v1",
        "api_key": "none",
        "models": [
            "local-model",   # llama-server uses whatever model is loaded
        ],
    },
}

API_KEY  = PROVIDERS["nvidia"]["api_key"]   # overridden by --provider
BASE_URL = PROVIDERS["nvidia"]["base_url"]  # overridden by --provider

FILE_TOOLS = {"Write", "Edit", "Bash"}  # tools that actually change files


@dataclass
class RealToolResult:
    model: str
    tool_called: bool
    tool_name: str
    tool_name_is_file_op: bool
    args_valid_json: bool
    args_non_empty: bool
    latency_s: float
    verdict: str   # PASS / PARTIAL / FAIL
    error: str = ""


TIMEOUT_S = 60.0  # overridden by --timeout flag


def test_model(model: str, payload: dict) -> RealToolResult:
    p = dict(payload)
    p["model"] = model
    p["stream"] = False
    p["max_completion_tokens"] = min(p.get("max_completion_tokens", 512), 512)
    p.pop("stream_options", None)  # only valid when stream=True

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            json=p,
            headers=headers,
            timeout=TIMEOUT_S,
        )
        latency = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return RealToolResult(
            model=model,
            tool_called=False, tool_name="", tool_name_is_file_op=False,
            args_valid_json=False, args_non_empty=False,
            latency_s=time.perf_counter() - start,
            verdict="FAIL", error=repr(e)[:200],
        )

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []

    t_called = len(tool_calls) > 0
    t_name   = tool_calls[0]["function"]["name"] if t_called else ""
    t_is_file = t_name in FILE_TOOLS

    t_args_valid = False
    t_args_nonempty = False
    if t_called:
        raw = tool_calls[0]["function"].get("arguments", "")
        try:
            parsed = json.loads(raw)
            t_args_valid = True
            t_args_nonempty = bool(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    if t_called and t_is_file and t_args_valid and t_args_nonempty:
        verdict = "PASS"
    elif t_called and t_args_valid and t_args_nonempty:
        verdict = "PARTIAL"   # called a tool but not a file op
    elif t_called:
        verdict = "PARTIAL"   # called something but args empty/invalid
    else:
        # Check if it replied with text describing what it would do
        text = msg.get("content") or ""
        verdict = "FAIL"

    return RealToolResult(
        model=model,
        tool_called=t_called,
        tool_name=t_name,
        tool_name_is_file_op=t_is_file,
        args_valid_json=t_args_valid,
        args_non_empty=t_args_nonempty,
        latency_s=latency,
        verdict=verdict,
    )


def print_result(r: RealToolResult) -> None:
    color = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}.get(r.verdict, "")
    reset = "\033[0m"
    print(f"\n  Verdict:      {color}{r.verdict}{reset}")
    print(f"  Tool called:  {r.tool_called}  ({r.tool_name or 'none'})")
    print(f"  Is file op:   {r.tool_name_is_file_op}")
    print(f"  Args valid:   {r.args_valid_json}")
    print(f"  Args nonempty:{r.args_non_empty}")
    print(f"  Latency:      {r.latency_s:.2f}s")
    if r.error:
        print(f"  Error:        {r.error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixture_real_request.json")
    parser.add_argument("--csv", default="results/tool_results.csv")
    parser.add_argument("--provider", default="nvidia", choices=list(PROVIDERS.keys()),
                        help="Which provider to test against")
    parser.add_argument("--models", nargs="+", help="Override model list")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="HTTP timeout in seconds (default 60; use 900 for local MoE models)")
    args = parser.parse_args()

    global API_KEY, BASE_URL, TIMEOUT_S
    TIMEOUT_S = args.timeout
    provider_cfg = PROVIDERS[args.provider]
    API_KEY  = provider_cfg["api_key"]
    BASE_URL = provider_cfg["base_url"]

    if not os.path.exists(args.fixture):
        raise SystemExit(f"Fixture not found: {args.fixture}  — run extract_fixture.py first")

    with open(args.fixture, encoding="utf-8") as f:
        payload = json.load(f)

    tool_count = len(payload.get("tools", []))
    print(f"Fixture loaded: {tool_count} tools in payload")
    print(f"Provider: {args.provider}  ({BASE_URL})")

    models = args.models if args.models else provider_cfg["models"]

    results = []
    for model in models:
        print(f"\n========== {model} ==========")
        r = test_model(model, payload)
        print_result(r)
        results.append(r)

    print("\n\n===== SUMMARY =====")
    for r in results:
        color = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}.get(r.verdict, "")
        reset = "\033[0m"
        print(f"  {color}{r.verdict:7}{reset}  {r.tool_name or 'no tool':20}  {r.model}")

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
