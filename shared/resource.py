# Stub for Unix-only `resource` module -- evalplus uses it to set memory limits.
# On Windows we just no-op everything; the sandbox still runs, just without
# memory limits (acceptable for a local benchmark).

RLIMIT_AS    = 0
RLIMIT_DATA  = 1
RLIMIT_STACK = 2
RLIMIT_CPU   = 3
RLIMIT_NOFILE = 4

def setrlimit(resource, limits):
    pass

def getrlimit(resource):
    return (-1, -1)
