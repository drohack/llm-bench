"""
Windows wrapper for evalplus.codegen -- patches signal.alarm before evalplus loads.
Usage: python evalplus_codegen_win.py <model> <dataset> [args...]
"""
import signal
import sys

# Patch Unix-only signal APIs before evalplus loads.
# SIGALRM / alarm() don't exist on Windows. Strategy:
#   - Set SIGALRM to a large int so multiprocess can negate it without crashing
#   - Make alarm() a no-op
#   - Wrap signal.signal() to silently swallow registrations for our fake SIGALRM
#     (catching the ValueError Windows raises for unknown signal numbers)
if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14  # same value as Linux; harmless on Windows if we swallow the error
    signal.alarm = lambda seconds: None
    _real_signal = signal.signal
    def _signal_win(sig, handler):
        try:
            return _real_signal(sig, handler)
        except (OSError, ValueError):
            return None   # Windows doesn't know SIGALRM -- silently ignore
    signal.signal = _signal_win

from evalplus.codegen import main
main()
