"""
Simple Windows-compatible evaluator for evalplus-generated solutions.
Runs prompt+solution code and test assertions directly without Unix signal/resource.
Usage: python evalplus_evaluate_win.py <samples.jsonl> <dataset>
"""
import json
import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def evaluate_solutions(samples_path: str, dataset: str = "humaneval") -> dict:
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    else:
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()

    samples = [json.loads(l) for l in open(samples_path, encoding="utf-8") if l.strip()]

    base_pass = 0
    plus_pass = 0
    total = len(samples)
    results = {}

    for sample in samples:
        task_id = sample["task_id"]
        solution = sample.get("solution", "")
        if task_id not in problems:
            continue

        problem = problems[task_id]
        code = problem["prompt"] + solution
        entry_point = problem["entry_point"]

        # Evaluate base tests
        base_ok = run_test(code, entry_point, problem["test"])
        # Evaluate plus tests (extended)
        plus_ok = run_test(code, entry_point, problem.get("plus_test", problem["test"]))

        results[task_id] = {
            "base_status": "pass" if base_ok else "fail",
            "plus_status": "pass" if plus_ok else "fail",
        }
        if base_ok:
            base_pass += 1
        if plus_ok:
            plus_pass += 1

    base_rate = base_pass / total if total else 0
    plus_rate = plus_pass / total if total else 0

    print(f"\nhumaneval (base tests)")
    print(f"pass@1:\t{base_rate:.3f}")
    print(f"\nhumaneval+ (base + extra tests)")
    print(f"pass@1:\t{plus_rate:.3f}")
    print(f"\nScore: {base_pass}/{total} base  {plus_pass}/{total} plus")

    return {"pass@1": base_rate, "base_pass@1": base_rate, "plus_pass@1": plus_rate}


def run_test(code: str, entry_point: str, test: str, timeout: float = 10.0) -> bool:
    """Run solution + test in a subprocess with a timeout to catch infinite loops."""
    import subprocess, sys, tempfile, os, json, base64

    # Encode the payload to avoid quoting issues
    payload = base64.b64encode(json.dumps({
        "code": code, "entry_point": entry_point, "test": test
    }).encode()).decode()

    runner = (
        "import sys, json, base64\n"
        "d = json.loads(base64.b64decode(sys.argv[1]))\n"
        "ns = {}\n"
        "exec(compile(d['code'], '<s>', 'exec'), ns)\n"
        "assert d['entry_point'] in ns\n"
        "exec(compile(d['test'], '<t>', 'exec'), ns)\n"
        "ns['check'](ns[d['entry_point']])\n"
        "print('PASS')\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(runner)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname, payload],
            capture_output=True, text=True, timeout=timeout
        )
        return "PASS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evalplus_evaluate_win.py <samples.jsonl> [humaneval|mbpp]")
        sys.exit(1)
    samples_path = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else "humaneval"
    evaluate_solutions(samples_path, dataset)
