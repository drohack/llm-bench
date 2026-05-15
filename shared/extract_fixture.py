"""
Extract the last complete OUTGOING REQUEST TO MODEL block from a clawgate debug log.
Saves it as bench/fixture_real_request.json for use by test_real_tools.py.
"""
import json
import os
import re
import sys

_HERE    = os.path.dirname(__file__)
OUT_PATH = os.path.join(_HERE, "fixture_real_request.json")

# Accept explicit path, or fall back to either log file name.
def _resolve_log(explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("clawgate-debug.log", "clawgate.log"):
        p = os.path.join(_HERE, "logs", name)
        if os.path.exists(p):
            return p
    raise SystemExit("No clawgate log found. Run with --log <path> or generate one with --debug.")

START = "--- [DEBUG] OUTGOING REQUEST TO MODEL ---"
END   = "---------------------------------------"

def extract_last_block(log_path: str) -> str:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Find all start positions
    starts = [m.start() for m in re.finditer(re.escape(START), content)]
    if not starts:
        raise SystemExit("No OUTGOING REQUEST blocks found in log.")

    # Walk backwards to find the last COMPLETE block (has a closing END after START)
    for start_pos in reversed(starts):
        json_start = content.index("\n", start_pos) + 1
        end_pos = content.find(END, json_start)
        if end_pos == -1:
            continue  # incomplete block, try previous
        raw = content[json_start:end_pos].strip()
        try:
            parsed = json.loads(raw)
            return parsed
        except json.JSONDecodeError:
            continue

    raise SystemExit("No valid JSON found in any OUTGOING REQUEST block.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="Path to clawgate debug log (auto-detected if omitted)")
    args = ap.parse_args()

    log_path = _resolve_log(args.log)
    print(f"Reading: {log_path}")
    payload = extract_last_block(log_path)

    tool_count = len(payload.get("tools", []))
    msg_count  = len(payload.get("messages", []))
    model      = payload.get("model", "unknown")
    print(f"Extracted: model={model}  messages={msg_count}  tools={tool_count}")

    # Swap in a fresh simple task as the last user message so the model
    # must call a file-writing tool — keeps the full tool list but forces action.
    task_msg = {
        "role": "user",
        "content": (
            "Create a file called hello.txt containing the text 'Hello from the model'. "
            "Use the Write tool to do this. Do not describe it — call the tool now."
        )
    }

    # Keep all system/context messages but replace the conversation tail
    messages = payload.get("messages", [])
    # Find last user message and replace from there
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), default=None
    )
    if last_user_idx is not None:
        messages = messages[:last_user_idx] + [task_msg]
    else:
        messages = messages + [task_msg]

    payload["messages"] = messages
    payload["max_completion_tokens"] = 1000
    payload["stream"] = True

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved fixture -> {OUT_PATH}  ({len(json.dumps(payload)) // 1024}KB)")
    print(f"Tool count in fixture: {tool_count}")


if __name__ == "__main__":
    main()
