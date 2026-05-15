import multiprocessing, sys

def test_child():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1000000, 1000000))
        print("resource stub OK in child")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    p = multiprocessing.Process(target=test_child)
    p.start()
    p.join(timeout=5)
    print("exit code:", p.exitcode)
