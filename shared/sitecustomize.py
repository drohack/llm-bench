# Loaded automatically by Python on startup when this directory is on PYTHONPATH.
# Patches Unix-only signal APIs so evalplus works on Windows -- applies to all
# spawned subprocesses (including evalplus's multiprocessing test sandbox).
import signal

if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14       # same int as Linux so multiprocess can negate it

if not hasattr(signal, 'ITIMER_REAL'):
    signal.ITIMER_REAL = 0    # evalplus uses this for per-test timeouts

if not hasattr(signal, 'alarm'):
    signal.alarm = lambda seconds: None

if not hasattr(signal, 'setitimer'):
    signal.setitimer = lambda which, seconds, interval=0: None

if not hasattr(signal, 'getitimer'):
    signal.getitimer = lambda which: (0, 0)

# Wrap signal.signal to silently swallow unknown signal registrations
_real_signal = signal.signal
def _safe_signal(sig, handler):
    try:
        return _real_signal(sig, handler)
    except (OSError, ValueError):
        return None
signal.signal = _safe_signal
