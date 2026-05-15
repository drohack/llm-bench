import multiprocessing

def check_signal():
    import signal
    has_alarm = hasattr(signal, 'alarm')
    print(f'Child has signal.alarm: {has_alarm}, SIGALRM={getattr(signal,"SIGALRM","missing")}')
    if has_alarm:
        try:
            signal.alarm(0)
            print('signal.alarm(0) works')
        except Exception as e:
            print(f'signal.alarm failed: {e}')

if __name__ == '__main__':
    p = multiprocessing.Process(target=check_signal)
    p.start()
    p.join()
    print(f'Child exit code: {p.exitcode}')
