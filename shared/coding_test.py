"""
Simple Python coding benchmark.

10 problems of increasing difficulty. The model must write a complete Python
function. Generated code is executed against test cases to determine pass/fail.

Usage:
    python coding_test.py --model my-model --base-url http://127.0.0.1:8081/v1 --api-key none
"""

import argparse
import ast
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict

from openai import OpenAI

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY  = "your-nvidia-api-key"

PROBLEMS = [
    {
        "name": "fizzbuzz",
        "difficulty": "easy",
        "prompt": (
            "Write a Python function `fizzbuzz(n)` that returns a list of strings for "
            "numbers 1 to n (inclusive). For multiples of 3 return 'Fizz', for multiples "
            "of 5 return 'Buzz', for multiples of both return 'FizzBuzz', otherwise return "
            "the number as a string."
        ),
        "tests": [
            "assert fizzbuzz(1) == ['1']",
            "assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']",
            "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
            "assert fizzbuzz(0) == []",
        ],
    },
    {
        "name": "is_palindrome",
        "difficulty": "easy",
        "prompt": (
            "Write a Python function `is_palindrome(s)` that returns True if the string s "
            "is a palindrome (reads the same forwards and backwards), ignoring case and "
            "non-alphanumeric characters. Return False otherwise."
        ),
        "tests": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('A man a plan a canal Panama') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('') == True",
        ],
    },
    {
        "name": "two_sum",
        "difficulty": "easy",
        "prompt": (
            "Write a Python function `two_sum(nums, target)` that takes a list of integers "
            "and a target integer. Return a list of two indices [i, j] (i < j) such that "
            "nums[i] + nums[j] == target. Assume exactly one solution exists."
        ),
        "tests": [
            "assert two_sum([2,7,11,15], 9) == [0,1]",
            "assert two_sum([3,2,4], 6) == [1,2]",
            "assert two_sum([3,3], 6) == [0,1]",
        ],
    },
    {
        "name": "flatten_list",
        "difficulty": "easy",
        "prompt": (
            "Write a Python function `flatten(lst)` that takes a nested list of arbitrary "
            "depth and returns a flat list of all values in order."
        ),
        "tests": [
            "assert flatten([1,[2,[3,[4]],5]]) == [1,2,3,4,5]",
            "assert flatten([]) == []",
            "assert flatten([[1,2],[3,4]]) == [1,2,3,4]",
            "assert flatten([1,2,3]) == [1,2,3]",
        ],
    },
    {
        "name": "binary_search",
        "difficulty": "medium",
        "prompt": (
            "Write a Python function `binary_search(arr, target)` that performs binary "
            "search on a sorted list. Return the index of target if found, or -1 if not."
        ),
        "tests": [
            "assert binary_search([1,3,5,7,9,11], 7) == 3",
            "assert binary_search([1,3,5,7,9,11], 6) == -1",
            "assert binary_search([], 1) == -1",
            "assert binary_search([1], 1) == 0",
        ],
    },
    {
        "name": "merge_intervals",
        "difficulty": "medium",
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that takes a list of "
            "[start, end] intervals and merges all overlapping intervals. Return the "
            "merged list sorted by start time."
        ),
        "tests": [
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
            "assert merge_intervals([[1,4]]) == [[1,4]]",
            "assert merge_intervals([]) == []",
        ],
    },
    {
        "name": "longest_common_prefix",
        "difficulty": "medium",
        "prompt": (
            "Write a Python function `longest_common_prefix(strs)` that finds the longest "
            "common prefix string among a list of strings. Return an empty string if there "
            "is no common prefix."
        ),
        "tests": [
            "assert longest_common_prefix(['flower','flow','flight']) == 'fl'",
            "assert longest_common_prefix(['dog','racecar','car']) == ''",
            "assert longest_common_prefix(['interview','interact','interface']) == 'inter'",
            "assert longest_common_prefix([]) == ''",
        ],
    },
    {
        "name": "valid_parentheses",
        "difficulty": "medium",
        "prompt": (
            "Write a Python function `is_valid(s)` that takes a string containing only "
            "'(', ')', '{', '}', '[', ']' and returns True if the brackets are valid "
            "(properly opened and closed in the correct order)."
        ),
        "tests": [
            "assert is_valid('()[]{}') == True",
            "assert is_valid('([)]') == False",
            "assert is_valid('{[]}') == True",
            "assert is_valid('') == True",
            "assert is_valid('(') == False",
        ],
    },
    {
        "name": "lru_cache",
        "difficulty": "hard",
        "prompt": (
            "Implement a Python class `LRUCache` with a fixed capacity. It must support:\n"
            "- `__init__(self, capacity)`: initialise with given capacity\n"
            "- `get(self, key)`: return value if key exists, else -1\n"
            "- `put(self, key, value)`: insert or update key. If capacity exceeded, "
            "evict the least recently used key."
        ),
        "tests": [
            "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1",
            "c = LRUCache(2); c.put(1,1); c.put(2,2); c.get(1); c.put(3,3); assert c.get(2)==-1",
            "c = LRUCache(1); c.put(1,1); c.put(2,2); assert c.get(1)==-1; assert c.get(2)==2",
        ],
    },
    {
        "name": "word_ladder",
        "difficulty": "hard",
        "prompt": (
            "Write a Python function `word_ladder(begin, end, word_list)` that finds the "
            "shortest transformation sequence from begin to end where each step changes "
            "exactly one letter and each intermediate word must be in word_list. Return "
            "the number of words in the shortest sequence, or 0 if none exists."
        ),
        "tests": [
            "assert word_ladder('hit','cog',['hot','dot','dog','lot','log','cog']) == 5",
            "assert word_ladder('hit','cog',['hot','dot','dog','lot','log']) == 0",
            "assert word_ladder('a','b',['b']) == 2",
        ],
    },
]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. When given a coding problem, respond with ONLY "
    "the Python function implementation. No explanation, no markdown, no preamble -- just "
    "the raw Python code starting with 'def ' or 'class '. The code must be complete and runnable."
)


def extract_code(text: str) -> str:
    """Pull code out of model response -- strips markdown fences if present."""
    # Remove ```python ... ``` fences
    text = re.sub(r"```(?:python)?\n?", "", text)
    text = text.replace("```", "").strip()
    return text


def run_code(code: str, tests: list[str], timeout: int = 10) -> tuple[bool, str]:
    """Execute code + tests in a subprocess. Returns (passed, error_message)."""
    test_block = "\n".join(tests)
    full = f"{code}\n\n{test_block}\nprint('__PASS__')\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                    encoding="utf-8") as f:
        f.write(full)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        if "__PASS__" in result.stdout:
            return True, ""
        err = (result.stderr or result.stdout or "no output").strip()
        return False, err[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, repr(e)
    finally:
        os.unlink(fname)


@dataclass
class CodingResult:
    model: str
    problem: str
    difficulty: str
    passed: bool
    latency_s: float
    error: str = ""
    generated_code: str = ""


def solve_problem(client: OpenAI, model: str, problem: dict,
                  timeout: int) -> CodingResult:
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": problem["prompt"]},
            ],
            max_tokens=800,
            temperature=0.1,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        raw_code = resp.choices[0].message.content or ""
        code = extract_code(raw_code)

        # Syntax check first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return CodingResult(model=model, problem=problem["name"],
                                difficulty=problem["difficulty"], passed=False,
                                latency_s=round(elapsed, 1),
                                error=f"SyntaxError: {e}", generated_code=code[:200])

        passed, err = run_code(code, problem["tests"])
        return CodingResult(model=model, problem=problem["name"],
                            difficulty=problem["difficulty"], passed=passed,
                            latency_s=round(elapsed, 1), error=err,
                            generated_code=code[:300])
    except Exception as e:
        elapsed = time.perf_counter() - start
        return CodingResult(model=model, problem=problem["name"],
                            difficulty=problem["difficulty"], passed=False,
                            latency_s=round(elapsed, 1), error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key",  default=None)
    ap.add_argument("--provider", choices=["local", "nvidia"], default="local")
    ap.add_argument("--timeout",  type=int, default=300)
    ap.add_argument("--csv",      default="results/coding_results.csv")
    args = ap.parse_args()

    if args.provider == "nvidia":
        base_url = args.base_url or NVIDIA_BASE
        api_key  = args.api_key  or NVIDIA_KEY
    else:
        base_url = args.base_url or "http://127.0.0.1:8081/v1"
        api_key  = args.api_key  or "none"

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"\nCoding benchmark -- {args.model}")
    print(f"{'Problem':<28} {'Diff':<8} {'Result':<8} {'Time':>7}")
    print("-" * 58)

    results = []
    for p in PROBLEMS:
        r = solve_problem(client, args.model, p, args.timeout)
        results.append(r)
        status = "PASS  " if r.passed else "FAIL  "
        print(f"{p['name']:<28} {p['difficulty']:<8} {status:<8} {r.latency_s:>6.1f}s")
        if not r.passed and r.error:
            print(f"  -> {r.error}")

    passed = sum(1 for r in results if r.passed)
    by_diff = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r.passed)

    print(f"\n{'?'*58}")
    print(f"Total: {passed}/{len(results)}", end="")
    for diff, vals in sorted(by_diff.items()):
        print(f"   {diff}: {sum(vals)}/{len(vals)}", end="")
    print()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        fields = [k for k in asdict(results[0]).keys() if k != "generated_code"]
        w = csv.DictWriter(f, fieldnames=fields + ["generated_code"])
        if f.tell() == 0:
            w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"Results -> {args.csv}")


if __name__ == "__main__":
    main()
