"""
EvalPlus (HumanEval+) wrapper for local llama-server benchmarking.

Runs code generation then evaluation in one shot, appends results to a CSV,
and prints a summary. Uses the OpenAI-compatible endpoint already used by
the rest of the bench suite.

Usage:
    python evalplus_test.py --model qwen3.6-35b-iq4xs --base-url http://127.0.0.1:8081/v1 --api-key none
    python evalplus_test.py --model qwen/qwen3.5-122b-a10b --provider nvidia
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Patch Unix-only signal APIs before any evalplus import, so multiprocess
# doesn't crash when iterating signal names during module load on Windows.
if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14   # use int (same as Linux) so multiprocess can negate it
    signal.alarm = lambda seconds: None
    _real_signal_fn = signal.signal
    def _signal_win(sig, handler):
        try:
            return _real_signal_fn(sig, handler)
        except (OSError, ValueError):
            return None
    signal.signal = _signal_win

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "local" / "results"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY  = "your-nvidia-api-key"
PYTHON      = sys.executable


def strip_function_to_body(solution: str, entry_point: str) -> str:
    """
    EvalPlus evaluates: prompt + solution.
    The prompt already has the function signature + docstring.
    If the model returned the full function, strip the signature so only
    the body remains -- otherwise we get a nested function and 0% pass rate.
    """
    import ast
    import textwrap
    if f"def {entry_point}" not in solution:
        return solution
    try:
        tree = ast.parse(solution)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                lines = solution.split("\n")
                # Skip def line + optional docstring, keep the real body
                body_nodes = [n for n in node.body
                              if not (isinstance(n, ast.Expr) and
                                      isinstance(getattr(n, "value", None),
                                                 ast.Constant))]
                if body_nodes:
                    start = body_nodes[0].lineno - 1
                elif node.body:
                    start = node.body[-1].lineno  # after docstring
                else:
                    return solution
                # Re-indent to 4 spaces: evalplus appends body after the
                # prompt (which ends with the closing docstring at 4-space
                # level), so the body must be at 4-space indent to be
                # inside the function.
                raw_body = "\n".join(lines[start:])
                dedented = textwrap.dedent(raw_body)
                return textwrap.indent(dedented, "    ")
    except Exception:
        pass
    return solution


def fix_samples_for_completion_mode(samples_path: Path, dataset: str) -> None:
    """Strip full function definitions down to body-only so EvalPlus evaluates correctly."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    problems = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()

    lines = samples_path.read_text(encoding="utf-8").splitlines()
    fixed = []
    changed = 0
    for line in lines:
        sample = json.loads(line)
        task_id = sample.get("task_id", "")
        if task_id in problems:
            ep = problems[task_id]["entry_point"]
            original = sample.get("solution", "")
            body = strip_function_to_body(original, ep)
            if body != original:
                sample["solution"] = body
                changed += 1
        fixed.append(json.dumps(sample))
    samples_path.write_text("\n".join(fixed), encoding="utf-8")
    print(f"  Fixed {changed}/{len(lines)} solutions (stripped function signatures for completion mode)")


def run_codegen(model: str, base_url: str, api_key: str,
                dataset: str, out_dir: Path, temperature: float) -> bool:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    # Use the Windows wrapper script instead of -m evalplus.codegen directly,
    # because signal.alarm (Unix-only) needs to be patched before evalplus imports.
    wrapper = HERE / "evalplus_codegen_win.py"
    cmd = [
        PYTHON, str(wrapper),
        model, dataset,
        "--backend",     "openai",
        "--base_url",    base_url,
        "--root",        str(out_dir),
        "--temperature", str(temperature),
        "--greedy",      "True",
        "--n_samples",   "1",
    ]
    print(f"Generating solutions ({dataset})...")
    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode == 0


def run_evaluate(samples_path: Path, dataset: str) -> dict:
    # Use our own evaluator instead of evalplus.evaluate which has deep
    # Unix-only dependencies (signal.setitimer, resource, etc.) that break
    # on Windows even with stubs due to multiprocessing spawn limitations.
    from evalplus_evaluate_win import evaluate_solutions
    print("Evaluating solutions...")
    return evaluate_solutions(str(samples_path), dataset)


def find_samples_file(out_dir: Path, model: str, dataset: str) -> Path | None:
    """EvalPlus saves to out_dir/<dataset>/<model>_openai_temp_*.jsonl (not raw)."""
    dataset_dir = out_dir / dataset
    model_slug = model.replace("/", "--")
    # Try exact model match first
    for f in sorted(dataset_dir.glob(f"{model_slug}*.jsonl")):
        if "raw" not in f.name and "eval_results" not in f.name:
            return f
    # Fallback: first non-raw jsonl
    for f in sorted(dataset_dir.glob("*.jsonl")):
        if "raw" not in f.name and "eval_results" not in f.name:
            return f
    return None


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True)
    ap.add_argument("--base-url",    default=None)
    ap.add_argument("--api-key",     default=None)
    ap.add_argument("--provider",    choices=["local", "nvidia"], default="local")
    ap.add_argument("--dataset",     default="humaneval",
                    help="humaneval or mbpp (default: humaneval)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--csv",         default=str(RESULTS_DIR / "evalplus_results.csv"))
    args = ap.parse_args()

    if args.provider == "nvidia":
        base_url = args.base_url or NVIDIA_BASE
        api_key  = args.api_key  or NVIDIA_KEY
    else:
        base_url = args.base_url or "http://127.0.0.1:8081/v1"
        api_key  = args.api_key  or "none"

    out_dir = HERE / "evalplus_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== EvalPlus ({args.dataset}) ===")
    print(f"Model:    {args.model}")
    print(f"Base URL: {base_url}")
    print()

    start = time.perf_counter()
    ok = run_codegen(args.model, base_url, api_key, args.dataset, out_dir, args.temperature)
    if not ok:
        print("ERROR: codegen step failed")
        sys.exit(1)

    samples = find_samples_file(out_dir, args.model, args.dataset)
    if not samples:
        print(f"ERROR: could not find samples file in {out_dir}")
        sys.exit(1)
    print(f"Samples: {samples}")

    # Strip full function defs down to body-only so EvalPlus evaluates correctly.
    # EvalPlus does prompt + solution; if solution includes the signature, the
    # function ends up nested inside the prompt's incomplete definition -> 0% pass.
    fix_samples_for_completion_mode(samples, args.dataset)

    scores = run_evaluate(samples, args.dataset)
    elapsed = time.perf_counter() - start

    pass1 = scores.get("pass@1") or scores.get("base_pass@1", 0.0)
    print(f"\nResult: pass@1 = {pass1:.1f}%  ({elapsed/60:.1f} min)")

    row = {
        "model":    args.model,
        "dataset":  args.dataset,
        "pass_at_1": pass1,
        "elapsed_min": round(elapsed / 60, 1),
        "samples_file": str(samples),
    }
    row.update({f"score_{k}": v for k, v in scores.items()})
    append_csv(Path(args.csv), row)
    print(f"Results -> {args.csv}")


if __name__ == "__main__":
    main()
