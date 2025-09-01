
from typing import List, Callable, Any

class Op:
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

def run_sequential(ops: List[Op]):
    out = None
    for op in ops:
        out = op.fn(*op.args, **op.kwargs)
    return out
