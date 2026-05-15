"""Simulate evalplus unsafe_execute to find the exact failure point."""
import sys, json, multiprocessing

def simulate_unsafe(code, entry_point, inp, expected):
    import contextlib, signal, tempfile, os, platform, faulthandler
    import traceback

    @contextlib.contextmanager
    def chdir(path):
        old = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old)

    errors = []

    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            # reliability_guard
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (500_000_000, 500_000_000))
            except Exception as e:
                errors.append(f"resource.setrlimit: {e}")

            try:
                faulthandler.disable()
            except Exception as e:
                errors.append(f"faulthandler.disable: {e}")

            # exec the code
            exec_globals = {}
            try:
                exec(code, exec_globals)
            except Exception as e:
                errors.append(f"exec(code): {e}")
                return False, errors

            if entry_point not in exec_globals:
                errors.append(f"entry_point '{entry_point}' not found in exec_globals")
                return False, errors

            fn = exec_globals[entry_point]

            # time_limit
            try:
                signal.setitimer(signal.ITIMER_REAL, 3.0)
                signal.signal(signal.SIGALRM, lambda s, f: None)
            except Exception as e:
                errors.append(f"time_limit setup: {e}")

            # run the function
            try:
                out = fn(*inp)
                errors.append(f"fn({inp}) = {out}, expected = {expected}, match = {out == expected}")
                return out == expected, errors
            except Exception as e:
                errors.append(f"fn call: {traceback.format_exc()}")
                return False, errors
            finally:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.path.insert(0, "C:/Users/droha/Workspace/SaltyChart-Claude/bench")
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    p = problems["HumanEval/0"]

    path = "evalplus_results/humaneval/qwen3.6-35b-iq4xs_openai_temp_0.0.jsonl"
    s = json.loads(open(path).readline())
    code = p["prompt"] + s["solution"]

    # Simulate with multiprocessing (spawned process like evalplus)
    mgr = multiprocessing.Manager()
    result = mgr.list()
    errs = mgr.list()

    def run_test(result, errs, code, ep, inp, exp):
        ok, errors = simulate_unsafe(code, ep, inp, exp)
        result.append(ok)
        errs.extend(errors)

    inp = ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)
    p_test = multiprocessing.Process(
        target=run_test,
        args=(result, errs, code, p["entry_point"], inp, True)
    )
    p_test.start()
    p_test.join(timeout=10)
    print("Exit code:", p_test.exitcode)
    print("Result:", list(result))
    print("Errors:")
    for e in errs:
        print(" ", e)
