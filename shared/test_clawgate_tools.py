"""
Test whether a model correctly handles Claude Code tool calls through clawgate.

Clawgate accepts Anthropic Messages API format and translates to OpenAI format
before forwarding to NVIDIA. This test sends requests exactly as Claude Code would.

Run this while clawgate is already running on port 8082.
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict

import anthropic

CLAWGATE_URL = "http://localhost:8082"
CLAWGATE_API_KEY = "dummy"  # clawgate in --mode=api doesn't auth the client

# Claude Code's actual tool definitions in Anthropic format
TOOLS = [
    {
        "name": "Read",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                }
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a shell command and return stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "description": {"type": "string", "description": "Short description of what the command does"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating it if it does not exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace old_string with new_string in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS}

TASK_PROMPT = (
    "You are a coding agent. A user reports that `src/app/page.tsx` has a syntax error. "
    "You MUST use the Read tool to inspect the file first before proposing any fix. "
    "Do not describe what you would do — call the tool now."
)

FAKE_READ_RESULT = """\
import React from 'react';
import { useState } from 'react';

export default function Page() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <h1>Count: {count}</h1>
      <button onClick={() => setCount(count + 1)>
        Increment
      </button>
    </div>
  );
}
"""

FAKE_RESULTS = {
    "Read": FAKE_READ_RESULT,
    "Write": "File written successfully.",
    "Edit": "Edit applied successfully.",
    "Bash": "$ exit 0",
}


@dataclass
class ClawgateTestResult:
    model: str
    turn1_tool_called: bool
    turn1_tool_name: str
    turn1_tool_name_valid: bool
    turn1_args_valid_json: bool
    turn1_args_complete: bool
    turn2_completed: bool
    turn2_response_type: str          # "tool", "text", or "error"
    turn1_latency_s: float
    turn2_latency_s: float
    verdict: str                      # PASS / PARTIAL / FAIL
    error: str = ""


def run_test(model: str, port: int) -> ClawgateTestResult:
    client = anthropic.Anthropic(
        base_url=f"http://localhost:{port}",
        api_key=CLAWGATE_API_KEY,
    )

    # --- Turn 1 ---
    start1 = time.perf_counter()
    try:
        resp1 = client.messages.create(
            model=model,
            max_tokens=512,
            system="You are a coding agent. Use tools whenever needed.",
            tools=TOOLS,
            messages=[{"role": "user", "content": TASK_PROMPT}],
        )
        lat1 = time.perf_counter() - start1
    except Exception as e:
        return ClawgateTestResult(
            model=model,
            turn1_tool_called=False, turn1_tool_name="", turn1_tool_name_valid=False,
            turn1_args_valid_json=False, turn1_args_complete=False,
            turn2_completed=False, turn2_response_type="error",
            turn1_latency_s=time.perf_counter() - start1, turn2_latency_s=0.0,
            verdict="FAIL", error=repr(e),
        )

    # Parse turn 1 response
    tool_use_blocks = [b for b in resp1.content if b.type == "tool_use"]
    t1_called = len(tool_use_blocks) > 0
    t1_name = tool_use_blocks[0].name if t1_called else ""
    t1_name_valid = t1_name in TOOL_NAMES if t1_name else False

    t1_args_valid = False
    t1_args_complete = False
    if t1_called:
        raw_input = tool_use_blocks[0].input
        if isinstance(raw_input, dict):
            t1_args_valid = True
            tool_def = next((t for t in TOOLS if t["name"] == t1_name), None)
            if tool_def:
                required = tool_def["input_schema"].get("required", [])
                t1_args_complete = all(k in raw_input for k in required)
        elif isinstance(raw_input, str):
            try:
                parsed = json.loads(raw_input)
                t1_args_valid = True
                tool_def = next((t for t in TOOLS if t["name"] == t1_name), None)
                if tool_def:
                    required = tool_def["input_schema"].get("required", [])
                    t1_args_complete = all(k in parsed for k in required)
            except json.JSONDecodeError:
                pass

    if not t1_called:
        text_content = " ".join(b.text for b in resp1.content if hasattr(b, "text"))[:120]
        return ClawgateTestResult(
            model=model,
            turn1_tool_called=False, turn1_tool_name="", turn1_tool_name_valid=False,
            turn1_args_valid_json=False, turn1_args_complete=False,
            turn2_completed=False, turn2_response_type="text",
            turn1_latency_s=lat1, turn2_latency_s=0.0,
            verdict="FAIL",
            error=f"Model replied with text: {text_content!r}",
        )

    # --- Turn 2: send fake tool result ---
    fake_content = FAKE_RESULTS.get(t1_name, "Done.")
    messages = [
        {"role": "user", "content": TASK_PROMPT},
        {"role": "assistant", "content": resp1.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_blocks[0].id,
                    "content": fake_content,
                }
            ],
        },
    ]

    start2 = time.perf_counter()
    try:
        resp2 = client.messages.create(
            model=model,
            max_tokens=512,
            system="You are a coding agent. Use tools whenever needed.",
            tools=TOOLS,
            messages=messages,
        )
        lat2 = time.perf_counter() - start2
    except Exception as e:
        verdict = "PARTIAL" if (t1_name_valid and t1_args_valid and t1_args_complete) else "FAIL"
        return ClawgateTestResult(
            model=model,
            turn1_tool_called=t1_called, turn1_tool_name=t1_name,
            turn1_tool_name_valid=t1_name_valid, turn1_args_valid_json=t1_args_valid,
            turn1_args_complete=t1_args_complete,
            turn2_completed=False, turn2_response_type="error",
            turn1_latency_s=lat1, turn2_latency_s=time.perf_counter() - start2,
            verdict=verdict, error=f"Turn 2: {repr(e)}",
        )

    turn2_tools = [b for b in resp2.content if b.type == "tool_use"]
    t2_type = "tool" if turn2_tools else "text"

    verdict = "PASS" if (t1_name_valid and t1_args_valid and t1_args_complete) else "PARTIAL"

    return ClawgateTestResult(
        model=model,
        turn1_tool_called=t1_called, turn1_tool_name=t1_name,
        turn1_tool_name_valid=t1_name_valid, turn1_args_valid_json=t1_args_valid,
        turn1_args_complete=t1_args_complete,
        turn2_completed=True, turn2_response_type=t2_type,
        turn1_latency_s=lat1, turn2_latency_s=lat2,
        verdict=verdict,
    )


def print_result(r: ClawgateTestResult) -> None:
    color = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}.get(r.verdict, "")
    reset = "\033[0m"
    print(f"\n  Verdict:       {color}{r.verdict}{reset}")
    print(f"  Tool called:   {r.turn1_tool_called}  ({r.turn1_tool_name or 'none'})")
    print(f"  Name valid:    {r.turn1_tool_name_valid}")
    print(f"  Args valid:    {r.turn1_args_valid_json}")
    print(f"  Args complete: {r.turn1_args_complete}")
    print(f"  Turn 2:        {r.turn2_response_type}  (completed={r.turn2_completed})")
    print(f"  Latency:       turn1={r.turn1_latency_s:.2f}s  turn2={r.turn2_latency_s:.2f}s")
    if r.error:
        print(f"  Error:         {r.error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--csv", default="results/tool_calling_results.csv")
    args = parser.parse_args()

    print(f"\n========== {args.model} ==========")
    result = run_test(args.model, args.port)
    print_result(result)

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))

    print(f"\n  Written to {args.csv}")


if __name__ == "__main__":
    main()
